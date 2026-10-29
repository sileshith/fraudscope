"""Delayed-label measurement, excluding unlabeled and not-yet-visible outcomes."""
import json
import numpy as np
from .service import connect
from .metrics import evaluate

def evaluate_feedback(db_path,as_of,lag_days=2):
    if as_of<0 or lag_days<0:raise ValueError("as_of and lag_days must be nonnegative")
    with connect(db_path) as db:
        eligible=db.execute("SELECT COUNT(*) FROM events WHERE event_time<=?",(as_of-lag_days*86400,)).fetchone()[0]
        rows=db.execute("SELECT e.amount,e.response,f.label FROM events e JOIN feedback f USING(transaction_id) WHERE e.event_time<=? AND f.available_time<=?",(as_of-lag_days*86400,as_of)).fetchall()
    response=[json.loads(r['response']) for r in rows]
    versions={r['model_version'] for r in response}
    result={"as_of":as_of,"maturity_lag_days":lag_days,"mature_events":eligible,"observed_labels":len(rows),
            "label_coverage":len(rows)/eligible if eligible else 0.,"versions":sorted(versions),
            "selection_bias_warning":"Observed outcomes may be selected by the review policy. Unlabeled transactions are excluded, never treated as legitimate.","by_model":{}}
    for version in sorted(versions):
        selected=[(r,p) for r,p in zip(rows,response) if p['model_version']==version]
        y=np.array([r['label'] for r,p in selected]);s=np.array([p['fraud_probability'] for r,p in selected]);a=np.array([r['amount'] for r,p in selected])
        result['by_model'][version]=evaluate(y,s,a) if len(np.unique(y))==2 else {"n":len(y),"status":"insufficient_class_variation"}
    return result

