import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import pytest
from fraudscope.synthetic import generate
from fraudscope.features import build_features
from fraudscope.contracts import NUMERIC,CATEGORICAL,RAW_COLUMNS
from fraudscope.pipeline import run_experiment

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp("training");df=generate(8000,19);x=build_features(df)
    result=run_experiment(df,x,NUMERIC,CATEGORICAL,root,19)
    return root,df,x,result

@pytest.fixture
def event():
    return {"transaction_id":"tx_1","event_time":100.,"account_id":"a1","device_id":"d1","amount":75.,
            "account_age_days":60.,"email_age_days":50.,"geo_distance_km":10.,"failed_logins_24h":0,
            "cross_border":0,"channel":"web","product":"retail","device_type":"desktop"}

@pytest.fixture
def api(trained,tmp_path):
    from fastapi.testclient import TestClient
    from fraudscope.service import create_app
    key="test-secret-with-more-than-24-characters"
    app=create_app(trained[0]/"model.joblib",tmp_path/"state.sqlite",key)
    with TestClient(app) as client:yield client,{"X-API-Key":key},tmp_path/"state.sqlite"

