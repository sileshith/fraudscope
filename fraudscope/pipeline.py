import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .contracts import NUMERIC,CATEGORICAL,FEATURE_VERSION
from .features import build_features
from .synthetic import generate
from .splits import temporal_split
from .models import Bundle,make_models,raw_score,fit_calibrator,save_bundle
from .metrics import evaluate,select_policy,day_bootstrap,review_cost
from .monitor import reference_profile,drift_report
from .explain import evidence_brief,reference_sensitivity

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def run_experiment(df,x,numeric,categorical,out,seed=42,dataset="synthetic",time_col="event_time",label_col="is_fraud",amount_col="amount",lag_days=2):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    idx,split=temporal_split(df,time_col=time_col,lag_days=lag_days)
    y=df[label_col].to_numpy(int);amount=df[amount_col].to_numpy(float)
    for k,ix in idx.items():
        if len(np.unique(y[ix]))<2: raise ValueError(f"{k} has only one class")
    data_hash=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest()
    version=f"{dataset}-{seed}-{data_hash[:10]}"
    fitted={}; table=[]
    for name,model in make_models(numeric,categorical,seed).items():
        started=time.perf_counter();tr=idx["train"];ca=idx["calibration"];va=idx["validation"]
        model.fit(x.iloc[tr],y[tr] if name!="isolation_forest" else None)
        anomaly=name=="isolation_forest"
        raw=raw_score(model,x.iloc[va],anomaly)
        entry={"model":name,"fit_seconds":time.perf_counter()-started,
               "validation_raw_ap":float(average_precision_score(y[va],raw))}
        if not anomaly:
            calibrator=fit_calibrator(y[ca],raw_score(model,x.iloc[ca]))
            score=calibrator.predict(raw)
            policy=select_policy(y[va],score,amount[va])
            entry["validation"]=evaluate(y[va],score,amount[va],policy)
            fitted[name]=(model,calibrator,policy)
        else: fitted[name]=(model,None,None)
        table.append(entry)
    # Freeze choice and policy before touching final test outcomes.
    candidates=[r for r in table if "validation" in r]
    winner=min(candidates,key=lambda r:r["validation"]["estimated_cost"])["model"]
    model,calibrator,policy=fitted[winner]
    reference={c:float(x.iloc[idx["train"]][c].median()) for c in numeric}
    reference.update({c:str(x.iloc[idx["train"]][c].mode().iloc[0]) for c in categorical})
    bundle=Bundle(model,calibrator,list(x.columns),policy,reference,winner,version,dataset,
                  FEATURE_VERSION if dataset=="synthetic" else "ieee-schema-v1")
    write_json(out/"selection.json",{"winner":winner,"policy":policy,"validation_results":table,
               "selection_rule":"lowest estimated validation cost among supervised candidates","test_used_for_selection":False})
    te=idx["test"];p=bundle.predict(x.iloc[te]);metrics=evaluate(y[te],p,amount[te],policy)
    for row in table:
        m,c,pol=fitted[row["model"]];r=raw_score(m,x.iloc[te],row["model"]=="isolation_forest")
        row["test_raw_ap"]=float(average_precision_score(y[te],r))
        row["test"]=evaluate(y[te],c.predict(r) if c is not None else r,amount[te],pol)
        if c is None:
            row["test"].pop("brier",None);row["test"].pop("log_loss",None)
    profile=reference_profile(x.iloc[idx["train"]],numeric,categorical)
    drift=drift_report(profile,x.iloc[te]);write_json(out/"drift.json",drift)
    write_json(out/"reference_profile.json",profile)
    # Diagnostic permutation importance is measured on validation, not tuned on test.
    va=idx["validation"]; base=average_precision_score(y[va],bundle.predict(x.iloc[va]));rng=np.random.default_rng(seed)
    importance=[]
    for col in x.columns:
        deltas=[]
        for _ in range(3):
            perm=x.iloc[va].copy();perm[col]=rng.permutation(perm[col].to_numpy())
            deltas.append(base-average_precision_score(y[va],bundle.predict(perm)))
        importance.append({"feature":col,"ap_drop_mean":float(np.mean(deltas)),"ap_drop_std":float(np.std(deltas))})
    slices=[]
    for col in categorical:
        for value in sorted(x.iloc[te][col].dropna().unique()):
            mask=x.iloc[te][col].to_numpy()==value
            if mask.sum()>=30 and len(np.unique(y[te][mask]))==2:
                slices.append({"feature":col,"value":str(value),**evaluate(y[te][mask],p[mask],amount[te][mask],policy)})
    # Most recent train accounts vs new-to-train accounts is a temporal slice, not a grouped split.
    if dataset=="synthetic":
        known=set(df.iloc[idx["train"]].account_id)
        for status,mask in [("seen_in_train",df.iloc[te].account_id.isin(known).to_numpy()),("unseen_in_train",~df.iloc[te].account_id.isin(known).to_numpy())]:
            if mask.sum()>=20 and len(np.unique(y[te][mask]))==2:
                slices.append({"feature":"account_train_status","value":status,**evaluate(y[te][mask],p[mask],amount[te][mask],policy)})
    ci=day_bootstrap(y[te],p,df.iloc[te][time_col],seed=seed)
    result={"dataset":dataset,"seed":seed,"rows":len(df),"model_version":version,"feature_count":x.shape[1],
            "feature_names":list(x.columns),"selection_rule":"minimum validation estimated cost",
            "winner":winner,"split":split,"policy":policy,"test":metrics,"average_precision_day_bootstrap_95pct":ci,
            "models":table,"slices":slices,"importance":sorted(importance,key=lambda r:-r["ap_drop_mean"]),
            "data_sha256":data_hash,"source_sha256":source_hash(),"production_approved":False,
            "benchmark_status":"not_run_real_data_required" if dataset=="synthetic" else "local_temporal_benchmark_not_leaderboard"}
    write_json(out/"metrics.json",result)
    prediction=pd.DataFrame({"event_time":df.iloc[te][time_col].to_numpy(),"label":y[te],"amount":amount[te],"score":p})
    prediction.to_csv(out/"holdout_predictions.csv",index=False)
    sensitivity=[]
    for fee in [1.,3.,10.]:
        for recovery in [.3,.5,.8]:
            cost=review_cost(y[te],amount[te],p>=policy["threshold"],fee,policy["false_alarm_friction"],recovery)
            sensitivity.append({"review_fee":fee,"recovery_fraction":recovery,"fixed_threshold":policy["threshold"],
                                "estimated_cost":cost,"estimated_avoided_cost":float((y[te]*amount[te]).sum()-cost)})
    pd.DataFrame(sensitivity).to_csv(out/"cost_sensitivity.csv",index=False)
    if dataset=="synthetic":
        cases=[]
        for pos in np.argsort(-p,kind="stable")[:20]:
            row=x.iloc[te[pos]].to_dict();decision="review" if p[pos]>=policy["threshold"] else "pass"
            cases.append({"transaction_id":df.iloc[te[pos]].transaction_id,"amount":amount[te[pos]],
                          "observed_label_for_demo":int(y[te[pos]]),"brief":evidence_brief(row,p[pos],decision,version),
                          "reference_sensitivity":reference_sensitivity(bundle,row)})
        write_json(out/"cases.json",cases)
        x.iloc[te].to_csv(out/"holdout_features.csv.gz",index=False,compression="gzip")
    save_bundle(bundle,out/"model.joblib",{"features":list(x.columns),"data_sha256":data_hash,
                 "source_sha256":source_hash(),"split":split,"seed":seed,"production_approved":False})
    from .reporting import make_report
    make_report(out,result,prediction,drift)
    return result

def run_demo(out="reports/demo",n=60000,seeds=(42,73,101)):
    root=Path(out);results=[]
    for seed in seeds:
        print(f"Generating {n:,} synthetic transactions, seed {seed}",flush=True)
        df=generate(n,seed);x=build_features(df)
        target=root/f"seed_{seed}";target.mkdir(parents=True,exist_ok=True)
        df.to_csv(target/"synthetic_transactions.csv.gz",index=False,compression="gzip")
        result=run_experiment(df,x,NUMERIC,CATEGORICAL,target,seed)
        results.append(result)
        print(f"{result['winner']}: AP={result['test']['average_precision']:.4f}, ROC-AUC={result['test']['roc_auc']:.4f}",flush=True)
    summary={"dataset":"synthetic","seeds":list(seeds),"rows_per_seed":n,"total_generated":n*len(seeds),
             "note":"Independent simulated populations. This is not a confidence interval for real-world performance.",
             "runs":[{"seed":r["seed"],"winner":r["winner"],**r["test"]} for r in results]}
    for metric in ["average_precision","roc_auc","recall_at_5pct","estimated_savings"]:
        vals=[r["test"][metric] for r in results]
        summary[metric]={"mean":float(np.mean(vals)),"std":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.}
    write_json(root/"multi_seed_summary.json",summary)
    return summary

