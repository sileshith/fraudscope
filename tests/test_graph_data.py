import numpy as np
import pandas as pd
import pytest
from fraudscope.graph_data import build_temporal_graph,perturb_graph,ieee_entity_proxies

def events():
    return pd.DataFrame({'event_time':[1.,2.,2.,3.,4.,5.],
                         'account_id':['a','a','a','b','a','a'],
                         'device_id':['d','d','e','d','e','d'],'is_fraud':[0,1,0,1,0,0]})

def test_graph_strict_time_and_relation_semantics():
    df=events();g=build_temporal_graph(df,neighbors=3)
    assert g.neighbors[1,0,0]==0
    assert 1 not in g.neighbors[2,0] # same time never sees earlier row order
    for i in range(len(df)):
        for r,col in enumerate(g.relations):
            for j in g.neighbors[i,r]:
                if j>=0:
                    assert df.iloc[j][col]==df.iloc[i][col]
                    assert df.iloc[j].event_time<df.iloc[i].event_time

def test_missing_entities_do_not_link_and_recent_k():
    df=events();df['device_id']=None;g=build_temporal_graph(df,neighbors=2)
    assert np.all(g.neighbors[:,1]==-1)
    assert list(g.neighbors[5,0])==[4,2]

def test_labels_and_future_ids_cannot_change_earlier_graph():
    df=events();g=build_temporal_graph(df)
    changed=df.copy();changed['is_fraud']=1-df.is_fraud;changed.loc[4:,'device_id']='future'
    h=build_temporal_graph(changed)
    assert np.array_equal(g.neighbors[:4],h.neighbors[:4])

def test_direction_reversal_that_looks_forward_rejected():
    g=build_temporal_graph(events());g.neighbors[0,0,0]=5
    with pytest.raises(ValueError,match='Future'):g.validate()

def test_deletion_and_causal_rewiring():
    g=build_temporal_graph(events());rows=np.arange(3,6)
    deleted,meta=perturb_graph(g,rows,1.,'delete')
    assert (deleted.neighbors[rows]<0).all()
    assert np.array_equal(deleted.neighbors[:3],g.neighbors[:3])
    changed,meta=perturb_graph(g,rows,.5,'rewire');changed.validate()
    assert np.array_equal((changed.neighbors>=0).sum(axis=2),(g.neighbors>=0).sum(axis=2))

def test_ieee_proxy_missingness():
    df=pd.DataFrame({'TransactionDT':[1,2,3],'card1':[1,1,2],'card2':[3,3,None],'addr1':[5,5,6],
                     'DeviceInfo':['unknown','iPhone',None],'id_31':['unknown','Safari',None]})
    x=ieee_entity_proxies(df)
    assert x.account_proxy[0]==x.account_proxy[1]
    assert pd.isna(x.account_proxy[2]) and pd.isna(x.device_proxy[0])

def test_rewire_norm_counts_final_adjacency_not_edit_operations():
    df=pd.DataFrame({'event_time':np.arange(12,dtype=float),'account_id':'a','device_id':'d'})
    g=build_temporal_graph(df,neighbors=8)
    h,meta=perturb_graph(g,np.arange(9,12),1.,'rewire',seed=4)
    def dense(graph):
        out=np.zeros((2,12,12),dtype=int)
        for i in range(12):
            for r in range(2):
                out[r,i,graph.neighbors[i,r][graph.neighbors[i,r]>=0]]=1
        return out
    actual=int(np.square(dense(g)-dense(h)).sum())
    assert meta['binary_adjacency_changed_entries']==actual
    assert actual<2*meta['changed_edges'] # later edits restore some original edges

