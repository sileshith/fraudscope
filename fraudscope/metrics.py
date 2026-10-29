"""Ranking, calibrated probability and explicit business-assumption metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

def review_cost(y, amount, review, review_fee=3., friction=5., recovery=.8):
    y=np.asarray(y); a=np.asarray(amount); r=np.asarray(review,bool)
    # Reviewed fraud retains (1-recovery) of loss. All reviews incur analyst cost.
    return float(np.sum(y*a*(1-recovery*r)+review_fee*r+friction*r*(1-y)))

def select_policy(y, score, amount, capacity=.05, review_fee=3., friction=5., recovery=.8):
    if not 0 < capacity <= 1: raise ValueError("capacity must be in (0,1]")
    s=np.asarray(score)
    # Candidate thresholds on validation only, including no review.
    thresholds=np.r_[1.000001,np.unique(np.quantile(s,np.linspace(0,1,301)))]
    candidates=[]
    for threshold in thresholds:
        r=s>=threshold
        if r.mean() <= capacity+1e-12:
            candidates.append((review_cost(y,amount,r,review_fee,friction,recovery),float(threshold),float(r.mean())))
    cost,threshold,rate=min(candidates,key=lambda x:(x[0],-x[1]))
    return {"threshold":threshold,"capacity_target":capacity,"validation_review_rate":rate,
            "validation_cost":cost,"review_fee":review_fee,"false_alarm_friction":friction,
            "fraud_recovery_fraction":recovery}

def evaluate(y, score, amount, policy=None):
    y=np.asarray(y,int); s=np.asarray(score,float); a=np.asarray(amount,float)
    if len(np.unique(y))<2: raise ValueError("Both classes required for evaluation")
    if not np.isfinite(s).all(): raise ValueError("Nonfinite scores")
    k=max(1,int(np.floor(.05*len(y))))
    # Stable ties preserve chronological order. This is retrospective ranking.
    top=np.argsort(-s,kind="stable")[:k]
    result={"n":len(y),"fraud_count":int(y.sum()),"fraud_rate":float(y.mean()),
            "average_precision":float(average_precision_score(y,s)),"roc_auc":float(roc_auc_score(y,s)),
            "recall_at_5pct":float(y[top].sum()/y.sum()),"precision_at_5pct":float(y[top].mean()),
            "value_capture_at_5pct":float((y[top]*a[top]).sum()/max((y*a).sum(),1e-12))}
    if np.all((s>=0)&(s<=1)):
        result.update(brier=float(brier_score_loss(y,s)),log_loss=float(log_loss(y,np.clip(s,1e-8,1-1e-8))))
    if policy:
        r=s>=policy["threshold"]; tp=int((r*y).sum()); fp=int((r*(1-y)).sum())
        cost=review_cost(y,a,r,policy["review_fee"],policy["false_alarm_friction"],policy["fraud_recovery_fraction"])
        result.update(review_rate=float(r.mean()),review_count=int(r.sum()),true_reviews=tp,false_reviews=fp,
            policy_precision=float(tp/max(r.sum(),1)),policy_recall=float(tp/y.sum()),
            false_positive_rate=float(fp/(1-y).sum()),estimated_cost=cost,
            no_review_cost=float((y*a).sum()),estimated_savings=float((y*a).sum()-cost),
            capacity_exceeded=bool(r.mean()>policy["capacity_target"]))
    return result

def day_bootstrap(y, score, times, repeats=300, seed=42):
    """Day-cluster percentile interval. Temporal dependence beyond one day remains."""
    rng=np.random.default_rng(seed); y=np.asarray(y); s=np.asarray(score)
    day=np.floor(np.asarray(times)/86400).astype(int); days=np.unique(day)
    groups={d:np.flatnonzero(day==d) for d in days}; values=[]
    for _ in range(repeats):
        ix=np.concatenate([groups[d] for d in rng.choice(days,len(days),replace=True)])
        if len(np.unique(y[ix]))==2: values.append(average_precision_score(y[ix],s[ix]))
    if not values: return {"low":None,"high":None,"replicates":0}
    return {"low":float(np.quantile(values,.025)),"high":float(np.quantile(values,.975)),"replicates":len(values)}

