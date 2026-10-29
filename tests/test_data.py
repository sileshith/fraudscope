import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from fraudscope.synthetic import generate
from fraudscope.contracts import Transaction,FEATURES
from fraudscope.features import build_features,extract
from fraudscope.splits import temporal_split

def test_generator_reproducible():
    pd.testing.assert_frame_equal(generate(1000,7),generate(1000,7))

def test_generator_contract():
    df=generate(1200,42)
    assert df.transaction_id.is_unique and df.event_time.is_monotonic_increasing
    assert 0<df.is_fraud.mean()<.15
    assert (df.label_available_time>df.event_time).all()
    for e in df.head(20).to_dict('records'):
        Transaction.model_validate({k:v for k,v in e.items() if k in Transaction.model_fields})

@pytest.mark.parametrize("bad",[{"amount":0},{"amount":float('nan')},{"amount":float('inf')},{"is_fraud":1},{"channel":"<script>"},{"transaction_id":"../../x"},{"failed_logins_24h":-1}])
def test_contract_rejects_invalid(event,bad):
    with pytest.raises(ValidationError):Transaction(**(event|bad))

def test_features_cannot_see_future_or_labels():
    df=generate(1200,42);a=build_features(df)
    mutated=df.copy();mutated.loc[600:,'amount']=9999;mutated['is_fraud']=1-df.is_fraud
    b=build_features(mutated)
    pd.testing.assert_frame_equal(a.iloc[:600],b.iloc[:600])
    assert set(a)==set(FEATURES)
    assert not {'is_fraud','scenario_audit_only','transaction_id','label_available_time'}&set(a)

def test_cold_start_and_strict_boundaries(event):
    f=extract(event,[],[])
    assert f['amount_ratio_7d']==1 and f['account_txn_7d']==0 and f['new_device_7d']==1
    f=extract(event,[(100,1,'d2'),(99,25,'d1'),(100-7*86400,1,'d2')],[(100,'a2'),(99,'a3')])
    assert f['amount_ratio_7d']==3 and f['account_txn_7d']==1 and f['device_accounts_24h']==1

def test_order_and_duplicates_rejected():
    df=generate(1000,42)
    with pytest.raises(ValueError):build_features(df.iloc[::-1])
    df.loc[1,'transaction_id']=df.loc[0,'transaction_id']
    with pytest.raises(ValueError):build_features(df)

def test_temporal_split_maturity_and_disjointness():
    df=generate(5000,42);ix,meta=temporal_split(df)
    assert len(np.unique(np.concatenate(list(ix.values()))))==sum(map(len,ix.values()))
    for stage,bound in zip(['train','calibration','validation'],meta['time_boundaries']):
        assert (df.iloc[ix[stage]].label_available_time<bound).all()
    assert df.iloc[ix['train']].event_time.max()<df.iloc[ix['calibration']].event_time.min()
    assert df.iloc[ix['validation']].event_time.max()<df.iloc[ix['test']].event_time.min()

