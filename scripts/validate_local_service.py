"""Exercise the actual ASGI application in-process, with a temporary state store.

This is not a live socket, load, or production latency test.
"""
from pathlib import Path
import json
import secrets
import tempfile
import time
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from fraudscope.service import create_app,verify_audit
from fraudscope.features import build_features
from fraudscope.models import load_bundle
from fraudscope.feedback import evaluate_feedback

root=Path(__file__).resolve().parents[1]
events=json.loads((root/'data/sample_events.json').read_text())
model=root/'reports/demo/seed_42/model.joblib'
expected=load_bundle(model).predict(build_features(pd.DataFrame(events)))
key=secrets.token_urlsafe(32);headers={'X-API-Key':key};latencies=[];probabilities=[]
with tempfile.TemporaryDirectory(prefix='fraudscope-validation-') as folder:
    db=Path(folder)/'state.sqlite'
    with TestClient(create_app(model,db,key)) as client:
        for event in events:
            started=time.perf_counter();response=client.post('/v1/score',headers=headers,json=event)
            latencies.append((time.perf_counter()-started)*1000)
            response.raise_for_status();probabilities.append(response.json()['fraud_probability'])
        assert np.allclose(expected,probabilities)
        for event in events[:10]:client.post('/v1/score',headers=headers,json=event).raise_for_status()
        assert client.get('/v1/operations',headers=headers).json()['events']==len(events)
        assert verify_audit(db)['valid']
        labels=pd.read_csv(root/'reports/demo/seed_42/synthetic_transactions.csv.gz',nrows=50)
        for row in labels.to_dict('records'):
            response=client.post('/v1/feedback',headers=headers,json={
                'transaction_id':row['transaction_id'],'label':row['is_fraud'],
                'available_time':row['label_available_time'],
                'source':'confirmed_chargeback' if row['is_fraud'] else 'confirmed_legitimate'})
            response.raise_for_status()
        mature=evaluate_feedback(db,events[-1]['event_time']+2*86400+1)
        assert mature['observed_labels']==50 and mature['label_coverage']==.25
        result={'transport':'in-process ASGI TestClient, not network HTTP','events':len(events),
                'offline_online_probabilities_match':True,'duplicate_retries':10,'stored_events_after_retries':len(events),
                'audit_valid':True,'feedback_records':50,'mature_label_coverage':mature['label_coverage'],
                'median_request_ms':float(np.median(latencies)),'p95_request_ms':float(np.percentile(latencies,95)),
                'note':'Sequential local correctness exercise. No throughput, availability or production latency guarantee.'}
(root/'reports/service-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
