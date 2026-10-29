"""Exercise the optional IEEE-shaped GNN route without real benchmark data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import pandas as pd
from fraudscope.benchmark import load_ieee
from fraudscope.graph_experiment import run_graph_experiment

root=Path(__file__).resolve().parents[1]
with TemporaryDirectory() as folder:
    path=Path(folder);n=3000;rng=np.random.default_rng(5)
    pd.DataFrame({'TransactionID':np.arange(n),'TransactionDT':np.linspace(0,120*86400,n),
        'TransactionAmt':rng.uniform(10,1000,n),'ProductCD':rng.choice(['W','C'],n),
        'card1':rng.integers(1,10,n),'card2':101,'addr1':110,
        'C1':rng.exponential(3,n),'V1':rng.normal(size=n),
        'isFraud':(rng.random(n)<.08).astype(int)}).to_csv(path/'train_transaction.csv',index=False)
    pd.DataFrame({'TransactionID':np.arange(0,n,2),'DeviceType':'desktop',
        'DeviceInfo':rng.choice(['device-a','device-b','device-c'],n//2),
        'id_31':'fixture-browser'}).to_csv(path/'train_identity.csv',index=False)
    df,x,num,cat=load_ieee(path)
    result=run_graph_experiment(df,x,num,cat,path/'output',epochs=1,dataset='ieee_cis_schema_fixture',
        time_col='TransactionDT',label_col='isFraud',amount_col='TransactionAmt')
    assert result['directed_edges']>0
    assert len(result['gnn_results'])==4
    assert not result['serving_model_changed']
    record={'status':'passed','data':'artificial IEEE-shaped fixture, not real IEEE-CIS',
        'rows':n,'epochs':1,'models':4,'directed_edges':result['directed_edges'],
        'verified':['identity left join','weak proxy graph construction','four model fits',
                    'calibration and policy','checkpoint replay','structural stress','report output'],
        'real_ieee_benchmark':'not_run','production_approved':False,'source_sha256':result['source_sha256']}
    (root/'reports/ieee-gnn-fixture-validation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
