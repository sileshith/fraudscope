import numpy as np
import pandas as pd

def reference_profile(x,numeric,categorical):
    result={"numeric":{},"categorical":{},"row_count":len(x)}
    for c in numeric:
        v=pd.to_numeric(x[c],errors="coerce").to_numpy(float); clean=v[np.isfinite(v)]
        edges=np.unique(np.quantile(clean,np.linspace(0,1,11))) if len(clean) else np.array([0.])
        edges=np.r_[-np.inf,edges[1:-1],np.inf]
        counts=np.r_[np.histogram(clean,bins=edges)[0],(~np.isfinite(v)).sum()]
        result["numeric"][c]={"edges":[None if not np.isfinite(z) else float(z) for z in edges],"counts":counts.tolist()}
    for c in categorical:
        result["categorical"][c]=x[c].fillna("__missing__").astype(str).value_counts().to_dict()
    return result

def _psi(a,b):
    a=(np.asarray(a,float)+.5); b=(np.asarray(b,float)+.5); a/=a.sum(); b/=b.sum()
    return float(np.sum((b-a)*np.log(b/a)))

def drift_report(profile,x):
    results=[]
    for c,p in profile["numeric"].items():
        edges=np.array([-np.inf]+p["edges"][1:-1]+[np.inf]); v=pd.to_numeric(x[c],errors="coerce").to_numpy(float)
        counts=np.r_[np.histogram(v[np.isfinite(v)],bins=edges)[0],(~np.isfinite(v)).sum()]
        value=_psi(p["counts"],counts)
        results.append({"feature":c,"psi":value,"missing_rate":float((~np.isfinite(v)).mean()),"alert":value>.2})
    for c,p in profile["categorical"].items():
        curr=x[c].fillna("__missing__").astype(str).value_counts().to_dict(); keys=sorted(set(p)|set(curr))
        value=_psi([p.get(k,0) for k in keys],[curr.get(k,0) for k in keys])
        results.append({"feature":c,"psi":value,"unknown_rate":float(sum(v for k,v in curr.items() if k not in p)/len(x)),"alert":value>.2})
    return {"current_rows":len(x),"reference_rows":profile["row_count"],"heuristic_psi_threshold":.2,
            "status":"investigate" if any(r["alert"] for r in results) else "no_alert",
            "features":sorted(results,key=lambda r:-r["psi"]),
            "note":"PSI is a heuristic. Input drift is not proof of fraud or model degradation. Never auto-retrain from this alert."}

