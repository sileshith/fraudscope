import json
import sqlite3
import numpy as np
import pandas as pd
import pytest
from fraudscope.features import build_features
from fraudscope.models import load_bundle
from fraudscope.service import create_app,verify_audit
from fastapi.testclient import TestClient

def test_authentication(api,event):
    c,h,db=api
    assert c.get('/healthz').status_code==200
    assert c.post('/v1/score',json=event).status_code==401
    assert c.post('/v1/score',headers={'X-API-Key':'wrong'},json=event).status_code==401
    assert c.get('/readyz',headers=h).status_code==200

def test_missing_secret_fails_closed(trained,tmp_path,monkeypatch):
    monkeypatch.delenv('FRAUDSCOPE_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='API_KEY'):
        with TestClient(create_app(trained[0]/'model.joblib',tmp_path/'x.sqlite',api_key='short')):pass

def test_idempotency_conflict_and_order(api,event):
    c,h,db=api;a=c.post('/v1/score',json=event,headers=h)
    assert a.status_code==200
    assert c.post('/v1/score',json=event,headers=h).json()==a.json()
    assert c.post('/v1/score',json=event|{'amount':80},headers=h).status_code==409
    assert c.post('/v1/score',json=event|{'transaction_id':'older','event_time':99},headers=h).status_code==409
    assert c.get('/v1/operations',headers=h).json()['events']==1

def test_online_offline_parity(api,event,trained):
    c,h,db=api
    events=[event,event|{'transaction_id':'tx_2','event_time':120.,'amount':150.},event|{'transaction_id':'tx_3','event_time':120.,'account_id':'a2'}]
    expected=build_features(pd.DataFrame(events));bundle=load_bundle(trained[0]/'model.joblib')
    actual=[]
    for row in events:
        response=c.post('/v1/score',headers=h,json=row);assert response.status_code==200
        actual.append(response.json()['fraud_probability'])
    assert np.allclose(actual,bundle.predict(expected))
    with sqlite3.connect(db) as con:stored=pd.DataFrame([json.loads(r[0]) for r in con.execute('SELECT features FROM events ORDER BY seq')])[expected.columns]
    pd.testing.assert_frame_equal(stored,expected,check_dtype=False)

def test_restart_retains_history(api,event,trained):
    c,h,db=api;c.post('/v1/score',headers=h,json=event)
    with TestClient(create_app(trained[0]/'model.joblib',db,h['X-API-Key'])) as new:
        assert new.get('/v1/operations',headers=h).json()['events']==1
        assert new.post('/v1/score',headers=h,json=event).status_code==200

def test_feedback_and_audit(api,event):
    c,h,db=api;c.post('/v1/score',headers=h,json=event)
    f={'transaction_id':'tx_1','label':1,'available_time':200000.,'source':'confirmed_chargeback'}
    assert c.post('/v1/feedback',headers=h,json=f).status_code==200
    assert c.post('/v1/feedback',headers=h,json=f).status_code==200
    assert c.post('/v1/feedback',headers=h,json=f|{'label':0,'source':'confirmed_legitimate'}).status_code==409
    assert c.post('/v1/feedback',headers=h,json=f|{'available_time':50.}).status_code==422
    assert c.get('/v1/audit',headers=h).json()['valid']
    with sqlite3.connect(db) as con:con.execute("UPDATE events SET response='{}'")
    assert verify_audit(db)['valid'] is False

def test_invalid_body_and_no_autonomous_action(api,event):
    c,h,db=api
    assert c.post('/v1/score',headers=h,json=event|{'is_fraud':1}).status_code==422
    r=c.post('/v1/score',headers=h,json=event).json()
    assert r['automatic_block'] is False and r['brief']['external_model_calls']==0
    assert c.post('/v1/feedback',headers=h,json={'transaction_id':'missing','label':1,'available_time':200.,'source':'confirmed_chargeback'}).status_code==404

def test_audit_detects_feature_state_change(api,event):
    c,h,db=api;c.post('/v1/score',headers=h,json=event)
    with sqlite3.connect(db) as con:con.execute('UPDATE events SET amount=999')
    assert verify_audit(db)['valid'] is False

def test_concurrent_duplicate_requests_are_atomic(api,event):
    from concurrent.futures import ThreadPoolExecutor
    c,h,db=api
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:c.post('/v1/score',headers=h,json=event),range(8)))
    assert all(r.status_code==200 for r in results)
    assert len({json.dumps(r.json(),sort_keys=True) for r in results})==1
    assert c.get('/v1/operations',headers=h).json()['events']==1

