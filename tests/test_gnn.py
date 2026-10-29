import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(2)
from fraudscope.gnn import CausalRelationalGNN,tensors,train,predict,save_weights,load_weights
from fraudscope.graph_data import build_temporal_graph,perturb_graph
from fraudscope.graph_teacher import causal_teacher_scores
from fraudscope.synthetic import generate
from fraudscope.features import build_features
from fraudscope.contracts import NUMERIC,CATEGORICAL
from fraudscope.splits import temporal_split

def tiny():
    rng=np.random.default_rng(6)
    df=pd.DataFrame({'event_time':np.arange(80,dtype=float),'account_id':['a']*80,'device_id':['d']*80})
    g=build_temporal_graph(df,neighbors=3);x=rng.normal(size=(80,6)).astype(np.float32);risk=np.linspace(.01,.8,80,dtype=np.float32)
    return x,g,risk

@pytest.mark.parametrize('mode',['no_edge_mlp','relational_sage','temporal_attention','risk_gated_attention'])
def test_graph_models_finite_cold_start_and_serialization(mode,tmp_path):
    x,g,r=tiny();torch.manual_seed(1);m=CausalRelationalGNN(6,mode=mode);d=tensors(x,g,r)
    p=predict(m,d,np.arange(80));assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    path=tmp_path/'gnn.pt';save_weights(m,path);restored=load_weights(path)
    assert np.allclose(p,predict(restored,d,np.arange(80)))

def test_no_edge_ablation_is_graph_invariant():
    x,g,r=tiny();m=CausalRelationalGNN(6,mode='no_edge_mlp');d=tensors(x,g,r)
    removed,_=perturb_graph(g,np.arange(80),1,'delete')
    assert np.allclose(predict(m,d,np.arange(80)),predict(m,tensors(x,removed,r),np.arange(80)))

def test_graph_changes_reach_model_output():
    x,g,r=tiny();torch.manual_seed(2);m=CausalRelationalGNN(6,mode='temporal_attention')
    removed,_=perturb_graph(g,np.arange(80),1,'delete')
    assert not np.allclose(predict(m,tensors(x,g,r),np.arange(80)),predict(m,tensors(x,removed,r),np.arange(80)))

def test_training_never_uses_nontraining_labels():
    x,g,r=tiny();d=tensors(x,g,r);y=np.tile([0,1],40);changed=y.copy();changed[40:]=1-changed[40:]
    torch.manual_seed(9);a=CausalRelationalGNN(6);torch.manual_seed(9);b=CausalRelationalGNN(6)
    train(a,d,y,np.arange(40),epochs=2,seed=7);train(b,d,changed,np.arange(40),epochs=2,seed=7)
    assert np.array_equal(predict(a,d,np.arange(80)),predict(b,d,np.arange(80)))

def test_teacher_labels_are_strictly_historical():
    df=generate(1800,9);x=build_features(df);ix,split=temporal_split(df)
    a,manifest,_=causal_teacher_scores(df,x,NUMERIC,CATEGORICAL,ix['train'],split['time_boundaries'][0])
    changed=df.copy();future=np.ones(len(df),bool);future[ix['train']]=False
    changed.loc[future,'is_fraud']=1-changed.loc[future,'is_fraud']
    b,_,_=causal_teacher_scores(changed,x,NUMERIC,CATEGORICAL,ix['train'],split['time_boundaries'][0])
    assert np.array_equal(a,b)
    assert all(v['max_fit_label_time'] is None or v['max_fit_label_time']<v['score_start'] for v in manifest)
    assert manifest[0]['method']=='fixed_prior_0.02_no_fitted_teacher'

def test_risk_gate_has_nonzero_gradient():
    x,g,r=tiny();m=CausalRelationalGNN(6);data=tensors(x,g,r)
    from fraudscope.gnn import forward_rows
    logits=forward_rows(m,data,np.arange(1,60));logits.sum().backward()
    assert m.gate_scale.grad is not None and m.gate_scale.grad.abs().sum()>0

