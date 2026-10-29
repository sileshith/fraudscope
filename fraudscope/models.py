from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression

def preprocessing(numeric,categorical):
    return ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True,add_indicator=True)),("scale",StandardScaler())]),numeric),
        ("category",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="unknown",keep_empty_features=True)),
                              ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=10,max_categories=32))]),categorical)
    ])

def make_models(numeric,categorical,seed=42):
    return {
        "logistic":Pipeline([("pre",preprocessing(numeric,categorical)),("model",LogisticRegression(C=.3,max_iter=800,random_state=seed))]),
        "hist_gradient_boosting":Pipeline([("pre",preprocessing(numeric,categorical)),("model",HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.07,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=seed))]),
        "isolation_forest":Pipeline([("pre",preprocessing(numeric,categorical)),("model",IsolationForest(n_estimators=120,max_samples=2048,random_state=seed,n_jobs=1))]),
    }

def raw_score(model,x,anomaly=False):
    if anomaly: return -model.score_samples(x)
    return model.predict_proba(x)[:,1]

@dataclass
class Bundle:
    model: object
    calibrator: object
    feature_names: list
    policy: dict
    reference: dict
    model_name: str
    model_version: str
    dataset: str
    feature_version: str

    def predict(self,x):
        if set(x.columns)!=set(self.feature_names):
            raise ValueError("Exact feature schema required")
        return self.calibrator.predict(raw_score(self.model,x[self.feature_names]))

def save_bundle(bundle,path,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,path,compress=3)
    manifest={**metadata,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "sklearn":sklearn.__version__,"python":platform.python_version(),
              "model_version":bundle.model_version,"dataset":bundle.dataset,
              "warning":"Only deserialize artifacts from a trusted source. Hash is integrity, not authenticity."}
    path.with_suffix(".json").write_text(json.dumps(manifest,indent=2))

def load_bundle(path):
    path=Path(path); meta=json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["sha256"]: raise ValueError("Model checksum mismatch")
    if meta["sklearn"]!=sklearn.__version__: raise ValueError("scikit-learn version mismatch. Retrain in this environment.")
    return joblib.load(path)

def fit_calibrator(y,score):
    # Held-out isotonic calibration. Saturation and small-sample risks are documented.
    return IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip").fit(score,y)

