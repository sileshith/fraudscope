"""Causal relation graphs: earlier transactions linked through shared entities.

This is a relation-typed transaction projection, not a full HGT with distinct
learned entity-node states. Missing identifiers never form a shared supernode.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class TemporalGraph:
    neighbors: np.ndarray
    age_hours: np.ndarray
    times: np.ndarray
    relations: list

    def validate(self):
        n,r,k=self.neighbors.shape
        if self.age_hours.shape!=(n,r,k) or len(self.times)!=n:raise ValueError("Graph shape mismatch")
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times)<0):raise ValueError("Invalid event times")
        if np.any((self.neighbors<-1)|(self.neighbors>=n)):raise ValueError("Invalid neighbor index")
        valid=self.neighbors>=0
        source=self.times[self.neighbors.clip(min=0)]
        if np.any(valid & (source>=self.times[:,None,None])):raise ValueError("Future or simultaneous edge rejected")
        expected=(self.times[:,None,None]-source)/3600
        if not np.isfinite(self.age_hours).all() or not np.allclose(self.age_hours[valid],expected[valid],atol=1e-4):
            raise ValueError("Edge ages disagree with event times")
        return self

def _key(value):
    if value is None or pd.isna(value):return None
    value=str(value)
    return None if value.lower() in {"", "nan", "unknown", "none"} else value

def build_temporal_graph(df,relations=("account_id","device_id"),time_col="event_time",neighbors=8,windows_days=(7,1)):
    if neighbors<1 or len(relations)!=len(windows_days) or any(w<=0 for w in windows_days):raise ValueError("Invalid graph config")
    t=df[time_col].to_numpy(float)
    if not np.isfinite(t).all() or np.any(np.diff(t)<0):raise ValueError("Sort events before graph construction")
    n=len(df);indices=np.full((n,len(relations),neighbors),-1,dtype=np.int64);ages=np.zeros_like(indices,dtype=np.float32)
    histories=[defaultdict(deque) for _ in relations]
    keys=[[ _key(v) for v in df[col]] for col in relations]
    start=0
    while start<n:
        end=start+1
        while end<n and t[end]==t[start]:end+=1
        for i in range(start,end):
            for r,window in enumerate(windows_days):
                key=keys[r][i]
                if key is None:continue
                q=histories[r][key]
                while q and t[q[0]]<=t[i]-window*86400:q.popleft()
                previous=list(q)[-neighbors:][::-1]
                indices[i,r,:len(previous)]=previous
                ages[i,r,:len(previous)]=(t[i]-t[previous])/3600
        # Group updates enforce strictly-earlier semantics for equal timestamps.
        for i in range(start,end):
            for r in range(len(relations)):
                key=keys[r][i]
                if key is not None:
                    q=histories[r][key];q.append(i)
                    while len(q)>neighbors:q.popleft()
        start=end
    return TemporalGraph(indices,ages,t,list(relations)).validate()

def perturb_graph(graph,rows,fraction,kind,seed=42):
    """Frozen-model graph-input stress, not a claim about real fraud behavior.

    Rewiring preserves target indegree where possible, but not source degree or
    true entity membership. Replacement sources are always strictly earlier.
    """
    if not 0<=fraction<=1 or kind not in {"delete","rewire"}:raise ValueError("Invalid perturbation")
    rng=np.random.default_rng(seed);adj=graph.neighbors.copy();age=graph.age_hours.copy()
    selected=np.zeros(adj.shape[0],bool);selected[np.asarray(rows)]=True
    positions=np.argwhere((adj>=0)&selected[:,None,None]);count=int(np.floor(len(positions)*fraction))
    changed=0
    if count:
        for i,r,k in positions[rng.choice(len(positions),count,replace=False)]:
            if kind=="delete":adj[i,r,k]=-1;age[i,r,k]=0;changed+=1
            else:
                last=int(np.searchsorted(graph.times,graph.times[i],side="left"))
                used=set(adj[i,r][adj[i,r]>=0]);candidate=None
                for _ in range(50):
                    j=int(rng.integers(0,max(last,1)))
                    if j<last and j not in used:candidate=j;break
                if candidate is not None:
                    adj[i,r,k]=candidate;age[i,r,k]=(graph.times[i]-graph.times[candidate])/3600;changed+=1
    # Count final binary adjacency differences, not sequential edit operations.
    # Rewiring can restore an original neighbor removed by an earlier edit.
    adjacency_changes=0
    for i in np.flatnonzero(selected):
        for r in range(adj.shape[1]):
            before=set(graph.neighbors[i,r][graph.neighbors[i,r]>=0])
            after=set(adj[i,r][adj[i,r]>=0])
            adjacency_changes+=len(before.symmetric_difference(after))
    return TemporalGraph(adj,age,graph.times,graph.relations).validate(),{"kind":kind,"requested_fraction":fraction,
            "eligible_edges":len(positions),"changed_edges":changed,"actual_fraction":changed/max(len(positions),1),
            "binary_adjacency_changed_entries":adjacency_changes}

def ieee_entity_proxies(df):
    """Weak relation keys from anonymized attributes, never verified identities."""
    out=pd.DataFrame({"event_time":df.TransactionDT.to_numpy()})
    def composite(columns):
        if not set(columns).issubset(df):return pd.Series([None]*len(df))
        valid=df[columns].notna().all(axis=1)
        for c in columns:valid &= ~df[c].astype(str).str.lower().isin(['','unknown','none','nan'])
        value=df[columns].astype(str).agg("|".join,axis=1)
        return value.where(valid,None)
    out["account_proxy"]=composite(["card1","card2","addr1"])
    out["device_proxy"]=composite(["DeviceInfo","id_31"])
    return out

