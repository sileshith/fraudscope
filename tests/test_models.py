import json
import numpy as np
import pandas as pd
import pytest
from fraudscope.models import load_bundle
from fraudscope.metrics import select_policy,evaluate,review_cost
from fraudscope.monitor import reference_profile,drift_report

def test_full_pipeline_artifacts(trained):
    root,df,x,r=trained
    for filename in ['model.joblib','model.json','metrics.json','selection.json','holdout_predictions.csv','portfolio.html','evaluation.png']:
        assert (root/filename).is_file()
    assert r['production_approved'] is False
    selection=json.loads((root/'selection.json').read_text())
    assert selection['test_used_for_selection'] is False
    assert r['test']['n']>1000
    assert 0<r['test']['average_precision']<=1
    assert r['policy']['validation_review_rate']<=.05

def test_serialization_and_unknown_categories(trained):
    root,df,x,r=trained;b=load_bundle(root/'model.joblib');a=b.predict(x.head(10));x2=x.head(10).copy();x2['product']='never_seen'
    assert np.isfinite(b.predict(x2)).all()
    assert np.allclose(a,load_bundle(root/'model.joblib').predict(x.head(10)))
    with pytest.raises(ValueError):b.predict(x.head().drop(columns=['product']))

def test_corrupt_model_rejected(trained,tmp_path):
    root=trained[0];p=tmp_path/'model.joblib';p.write_bytes((root/'model.joblib').read_bytes()+b'bad')
    p.with_suffix('.json').write_bytes((root/'model.json').read_bytes())
    with pytest.raises(ValueError,match='checksum'):load_bundle(p)

def test_cost_math():
    assert review_cost([1,0],[100,50],[1,1])==pytest.approx(31)
    assert review_cost([1,0],[100,50],[0,0])==100

def test_threshold_ties_and_capacity():
    p=select_policy([1]*10+[0]*90,[.5]*100,[100]*100,capacity=.05)
    assert p['threshold']>1 and p['validation_review_rate']==0

def test_ranking_metrics_and_single_class_error():
    y=np.array([1,0]*50);s=y*.9;result=evaluate(y,s,np.ones(100))
    assert result['average_precision']==1 and result['precision_at_5pct']==1
    with pytest.raises(ValueError):evaluate([0,0],[.1,.3],[1,1])

def test_drift_detects_shift():
    x=pd.DataFrame({'a':np.arange(1000),'b':['known']*1000})
    ref=reference_profile(x,['a'],['b'])
    assert drift_report(ref,x)['status']=='no_alert'
    shifted=x.copy();shifted['a']+=2000;shifted['b']='new'
    assert drift_report(ref,shifted)['status']=='investigate'

