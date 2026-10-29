"""Export trusted Python weights and independent parity cases for Worker inference.
Run with the original fraudscope package importable. No retraining occurs here.
"""
from pathlib import Path
import argparse,json,joblib,hashlib
import numpy as np
import pandas as pd
import torch
from fraudscope.features import build_features
from fraudscope.gnn import load_weights,tensors,predict,MODES
from fraudscope.graph_data import TemporalGraph
p=argparse.ArgumentParser();p.add_argument('--research-root',required=True);args=p.parse_args()
root=Path(args.research_root);out=Path(__file__).resolve().parents[1]/'model-data';out.mkdir(exist_ok=True)
r=root/'reports';gp=r/'gnn/seed_42';b=joblib.load(r/'demo/seed_42/model.joblib');g=joblib.load(gp/'preprocessing.joblib')
def pre(v):
 n=v.named_transformers_['numeric'];c=v.named_transformers_['category']['encode']
 assert len(n['impute'].indicator_.features_)==0
 assert all(x is None for x in c.infrequent_categories_)
 return {'numeric':list(v.transformers_[0][2]),'categorical':list(v.transformers_[1][2]),'median':n['impute'].statistics_.tolist(),'mean':n['scale'].mean_.tolist(),'scale':n['scale'].scale_.tolist(),'categories':[x.tolist() for x in c.categories_]}
def linear(m):return {'pre':pre(m['pre']),'coef':m['model'].coef_[0].tolist(),'intercept':float(m['model'].intercept_[0])}
def cal(c):return {'x':c.X_thresholds_.tolist(),'y':c.y_thresholds_.tolist()}
payload={'version':'fraudscope-edge-v2.0.0','dataset':'synthetic','active':{**linear(b.model),'calibration':cal(b.calibrator),'policy':b.policy,'original_version':b.model_version},'graph_pre':pre(g['preprocessor']),'teacher':linear(g['frozen_future_teacher']),'gnn':{}}
models={}
for name in MODES:
 m=load_weights(gp/f'{name}.pt');models[name]=m;c=joblib.load(gp/f'{name}_policy.joblib');payload['gnn'][name]={'config':m.config,'weights':{k:v.tolist() for k,v in m.state_dict().items()},'calibration':cal(c['calibrator']),'policy':c['policy']}
(out/'models.json').write_text(json.dumps(payload,separators=(',',':')))
df=pd.read_csv(r/'demo/seed_42/synthetic_transactions.csv.gz');x=build_features(df);encoded=g['preprocessor'].transform(x).astype(np.float32);a=np.load(gp/'causal_graph.npz');graph=TemporalGraph(a['neighbors'],a['age_hours'],a['times'],['account_id','device_id']);data=tensors(encoded,graph,a['risk']);rows=np.unique(np.r_[np.arange(8),np.linspace(48000,59999,72).astype(int)]);expected={name:predict(m,data,rows).tolist() for name,m in models.items()}
cases=[]
for pos,i in enumerate(rows):
 rel=[]
 for rid in range(2):
  rel.append([{'encoded':encoded[j].tolist(),'risk':float(a['risk'][j]),'age_hours':float(a['age_hours'][i,rid,k])} for k,j in enumerate(a['neighbors'][i,rid]) if j>=0])
 cases.append({'features':x.iloc[i].to_dict(),'encoded':encoded[i].tolist(),'neighbors':rel,'active_raw':float(b.model.predict_proba(x.iloc[[i]])[0,1]),'active_probability':float(b.predict(x.iloc[[i]])[0]),'teacher_risk':float(g['frozen_future_teacher'].predict_proba(x.iloc[[i]])[0,1]),'graph_raw':{name:expected[name][pos] for name in MODES},'graph_probability':{name:float(joblib.load(gp/f'{name}_policy.joblib')['calibrator'].predict([expected[name][pos]])[0]) for name in MODES}})
(out/'parity-fixtures.json').write_text(json.dumps(cases,separators=(',',':')))
summary=json.loads((r/'gnn/multi_seed_summary.json').read_text());summary['stress']=json.loads((gp/'structural_stress.json').read_text());summary['policy']=b.policy;summary['original_model_version']=b.model_version
(out/'evidence.json').write_text(json.dumps(summary,separators=(',',':')))
(out/'provenance.json').write_text(json.dumps({'export_version':'2.0.0','source_data_sha256':json.loads((gp/'metrics.json').read_text())['data_sha256'],'research_source_sha256':json.loads((gp/'metrics.json').read_text())['source_sha256'],'models_sha256':hashlib.sha256((out/'models.json').read_bytes()).hexdigest(),'parity_cases':len(cases),'real_ieee_benchmark':'not_run','live_role':'logistic decisions; risk-gated GNN shadow'},indent=2))
print('Exported five trained models and',len(cases),'independent parity cases')
