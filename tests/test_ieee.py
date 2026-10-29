import numpy as np
import pandas as pd
import pytest
from fraudscope.benchmark import load_ieee,run_ieee

def fixture_files(path,n=3000):
    rng=np.random.default_rng(5)
    x=pd.DataFrame({'TransactionID':np.arange(n),'TransactionDT':np.linspace(0,120*86400,n),
                    'TransactionAmt':rng.uniform(10,1000,n),'ProductCD':rng.choice(['W','C'],n),
                    'card1':rng.integers(1,10,n),'C1':rng.exponential(3,n),'V1':rng.normal(size=n),
                    'isFraud':(rng.random(n)<.08).astype(int),'target_leak':1})
    x.to_csv(path/'train_transaction.csv',index=False)
    pd.DataFrame({'TransactionID':np.arange(0,n,2),'DeviceType':['desktop']*((n+1)//2),'id_12':['Found']*((n+1)//2)}).to_csv(path/'train_identity.csv',index=False)
    return x

def test_ieee_left_join_allowlist_and_missing_identity(tmp_path):
    original=fixture_files(tmp_path);df,x,num,cat=load_ieee(tmp_path)
    assert len(df)==len(original) and df.DeviceType.isna().sum()==1500
    assert not {'isFraud','TransactionID','TransactionDT','target_leak'}&set(x)
    assert 'card1' in cat and 'TransactionAmt' in num

def test_ieee_duplicate_identity_rejected(tmp_path):
    fixture_files(tmp_path);p=tmp_path/'train_identity.csv';x=pd.read_csv(p);pd.concat([x,x.head(1)]).to_csv(p,index=False)
    with pytest.raises(ValueError):load_ieee(tmp_path)

def test_ieee_absent_data_clear_failure(tmp_path):
    with pytest.raises(FileNotFoundError,match='authorized'):load_ieee(tmp_path)

def test_ieee_pipeline_on_artificial_schema_fixture(tmp_path):
    fixture_files(tmp_path);result=run_ieee(tmp_path,tmp_path/'output')
    assert result['dataset']=='ieee_cis'
    assert result['benchmark_status']=='local_temporal_benchmark_not_leaderboard'
    assert result['test']['n']>500
    # This test verifies plumbing only. The fixture is not real IEEE-CIS data.

