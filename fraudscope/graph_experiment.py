"""Frozen-protocol GNN comparison on the same four temporal stages as v1."""
from pathlib import Path
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from .contracts import NUMERIC,CATEGORICAL
from .synthetic import generate
from .features import build_features
from .splits import temporal_split
from .models import preprocessing,fit_calibrator
from .metrics import evaluate,select_policy,day_bootstrap
from .pipeline import write_json,source_hash
from .graph_data import build_temporal_graph,perturb_graph,ieee_entity_proxies
from .graph_teacher import causal_teacher_scores
from .gnn import MODES,CausalRelationalGNN,tensors,train,predict,save_weights,load_weights

def run_graph_experiment(df,x,numeric,categorical,out,seed=42,epochs=12,neighbors=8,
                         reference=None,dataset='synthetic',time_col='event_time',label_col='is_fraud',amount_col='amount',lag_days=2):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    idx,split=temporal_split(df,time_col=time_col,lag_days=lag_days)
    y=df[label_col].to_numpy(int);amount=df[amount_col].to_numpy(float)
    if epochs<1 or epochs>500:raise ValueError('epochs must be between 1 and 500')
    for key,rows in idx.items():
        if len(np.unique(y[rows]))<2:raise ValueError(f'{key} requires both classes')
    data_hash=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest()
    config={'rows':len(df),'seed':seed,'epochs':epochs,'batch_size':512,'hidden':48,'heads':4,'neighbors_per_relation':neighbors,
            'relation_windows_days':[7,1],'models':list(MODES),'learning_rate':.001,'weight_decay':.001,
            'source_sha256':source_hash(),'data_sha256':data_hash,'selection':'minimum validation estimated cost',
            'protocol':'fixed epoch count; no test tuning; exploratory extension to an already inspected synthetic benchmark'}
    write_json(out/'protocol.json',config)
    if dataset=='synthetic':
        graph=build_temporal_graph(df,neighbors=neighbors)
    else:
        entities=ieee_entity_proxies(df)
        graph=build_temporal_graph(entities,relations=('account_proxy','device_proxy'),neighbors=neighbors)
    pre=preprocessing(numeric,categorical);pre.fit(x.iloc[idx['train']]);encoded=pre.transform(x).astype(np.float32)
    risk,teacher_manifest,teacher=causal_teacher_scores(df,x,numeric,categorical,idx['train'],split['time_boundaries'][0],seed,time_col,label_col,lag_days)
    write_json(out/'teacher_folds.json',teacher_manifest)
    joblib.dump({'preprocessor':pre,'frozen_future_teacher':teacher,'feature_names':list(x.columns)},out/'preprocessing.joblib',compress=3)
    np.savez_compressed(out/'causal_graph.npz',neighbors=graph.neighbors,age_hours=graph.age_hours,times=graph.times,risk=risk)
    data=tensors(encoded,graph,risk);fitted={};table=[];predictions={}
    tr=idx['train'];ca=idx['calibration'];va=idx['validation'];te=idx['test']
    for mode in MODES:
        print(f'GNN seed {seed}: {mode}, {epochs} fixed epochs',flush=True)
        torch.manual_seed(seed);model=CausalRelationalGNN(encoded.shape[1],len(graph.relations),mode=mode)
        started=time.perf_counter();history=train(model,data,y,tr,epochs=epochs,seed=seed)
        raw_ca=predict(model,data,ca);raw_va=predict(model,data,va)
        calibration=fit_calibrator(y[ca],raw_ca);pva=calibration.predict(raw_va);policy=select_policy(y[va],pva,amount[va])
        entry={'model':mode,'fit_seconds':time.perf_counter()-started,'validation_raw_ap':float(average_precision_score(y[va],raw_va)),
               'validation':evaluate(y[va],pva,amount[va],policy),'policy':policy,
               'parameters':sum(p.numel() for p in model.parameters())}
        fitted[mode]=(model,calibration,policy);table.append(entry)
        save_weights(model,out/f'{mode}.pt');joblib.dump({'calibrator':calibration,'policy':policy},out/f'{mode}_policy.joblib')
        write_json(out/f'{mode}_training.json',history)
    baseline=[]
    if reference is not None:
        reference=json.loads(Path(reference).read_text())
        if reference['data_sha256']!=data_hash:raise ValueError('Baseline data fingerprint mismatch')
        if reference['split']!=split:raise ValueError('Baseline split mismatch')
        baseline=reference['models']
    candidate_rows=table+[r for r in baseline if 'validation' in r]
    winner=min(candidate_rows,key=lambda r:r['validation']['estimated_cost'])['model']
    graph_winner=min([r for r in table if r['model']!='no_edge_mlp'],key=lambda r:r['validation']['estimated_cost'])['model']
    write_json(out/'selection.json',{'overall_candidate_winner':winner,'graph_candidate_winner':graph_winner,
                'validation_only':True,'serving_model_changed':False,'candidates':candidate_rows})
    # Final outcomes are accessed only after all choices above are frozen.
    for row in table:
        name=row['model'];model,cal,policy=fitted[name];raw=predict(model,data,te);p=cal.predict(raw);predictions[name]=p
        row['test_raw_ap']=float(average_precision_score(y[te],raw));row['test']=evaluate(y[te],p,amount[te],policy)
        row['ap_day_bootstrap_95pct']=day_bootstrap(y[te],p,df.iloc[te][time_col],seed=seed,repeats=200)
        restored=load_weights(out/f'{name}.pt')
        if not np.allclose(predict(restored,data,te[:30]),raw[:30]):raise RuntimeError('GNN checkpoint round-trip mismatch')
    model,cal,policy=fitted[graph_winner];clean_raw,clean_z=predict(model,data,te,embedding=True);clean=cal.predict(clean_raw)
    stress=[]
    for kind in ['delete','rewire']:
        for fraction in [.1,.3]:
            for replicate in [0,1,2]:
                perturbed,changes=perturb_graph(graph,te,fraction,kind,seed+replicate)
                raw,z=predict(model,tensors(encoded,perturbed,risk),te,embedding=True);p=cal.predict(raw)
                stress.append({**changes,'replicate':replicate,'model':graph_winner,'frozen_weights':True,
                    'raw_relation_adjacency_frobenius_delta':float(np.sqrt(changes['binary_adjacency_changed_entries'])),
                    'mean_embedding_l2_drift':float(np.linalg.norm(z-clean_z,axis=1).mean()),
                    'mean_absolute_score_drift':float(np.abs(p-clean).mean()),
                    'threshold_flip_rate':float(np.mean((p>=policy['threshold'])!=(clean>=policy['threshold']))),
                    **evaluate(y[te],p,amount[te],policy)})
    write_json(out/'structural_stress.json',stress)
    result={'dataset':dataset,'rows':len(df),'seed':seed,'split':split,'data_sha256':data_hash,'source_sha256':source_hash(),
            'config':config,'graph_relations':graph.relations,'directed_edges':int((graph.neighbors>=0).sum()),
            'fraction_without_history':float(np.mean((graph.neighbors<0).all(axis=(1,2)))),
            'overall_candidate_winner':winner,'graph_candidate_winner':graph_winner,'models':baseline+table,
            'gnn_results':table,'serving_model_changed':False,'production_approved':False,
            'benchmark_status':'real_ieee_not_run' if dataset=='synthetic' else 'local_temporal_ieee_only',
            'risk_gate_note':'Logistic teacher with five past-only training blocks and frozen training-only future scorer; not the published LightGBM method',
            'architecture_note':'One-hop typed transaction projection with four-head relative-time attention. No recurrent TGN memory or full HGT entity states.'}
    write_json(out/'metrics.json',result)
    pd.DataFrame({'row_index':te,'event_time':df.iloc[te][time_col].to_numpy(),'label':y[te],'amount':amount[te],**predictions}).to_csv(out/'holdout_predictions.csv',index=False)
    # Selected graph candidate case: labels are not saved as GNN inputs.
    pos=int(np.argmax(predictions[graph_winner]));row=int(te[pos]);case={'transaction_row':row,'score':float(predictions[graph_winner][pos]),'relations':[]}
    for relation,name in enumerate(graph.relations):
        valid=graph.neighbors[row,relation]>=0
        case['relations'].append({'relation':name,'historical_rows':graph.neighbors[row,relation,valid].tolist(),
                                  'age_hours':graph.age_hours[row,relation,valid].tolist(),
                                  'teacher_risk_at_historical_event':risk[graph.neighbors[row,relation,valid]].tolist()})
    write_json(out/'graph_case.json',case)
    from .graph_reporting import graph_report
    graph_report(out,result,stress)
    return result

def run_graph_demo(out='reports/gnn',rows=60000,seeds=(42,73,101),epochs=12,neighbors=8,baseline_root='reports/demo'):
    all_results=[]
    for seed in seeds:
        df=generate(rows,seed);x=build_features(df);reference=Path(baseline_root)/f'seed_{seed}/metrics.json'
        if not reference.exists():
            from .pipeline import run_experiment
            run_experiment(df,x,NUMERIC,CATEGORICAL,reference.parent,seed)
        all_results.append(run_graph_experiment(df,x,NUMERIC,CATEGORICAL,Path(out)/f'seed_{seed}',seed,epochs,neighbors,reference))
    summary={'dataset':'synthetic','rows_per_seed':rows,'seeds':list(seeds),'real_ieee_status':'not_run','models':[]}
    for name in [r['model'] for r in all_results[0]['models']]:
        observed=[next(r for r in result['models'] if r['model']==name) for result in all_results]
        entry={'model':name}
        for metric in ['average_precision','roc_auc','recall_at_5pct']:
            vals=[r['test'][metric] for r in observed]
            entry[metric+'_mean']=float(np.mean(vals));entry[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        summary['models'].append(entry)
    summary['selection_by_seed']=[{'seed':r['seed'],'overall':r['overall_candidate_winner'],'graph':r['graph_candidate_winner']} for r in all_results]
    write_json(Path(out)/'multi_seed_summary.json',summary)
    return summary

