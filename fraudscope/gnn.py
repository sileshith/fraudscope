"""Optional PyTorch causal relational GNNs. No PyG extension is required.

Original research-oriented implementation inspired by temporal message passing,
typed attention and risk gates. It does not reproduce TGN, HGT or RGH-GNN.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES=("no_edge_mlp","relational_sage","temporal_attention","risk_gated_attention")

class CausalRelationalGNN(nn.Module):
    def __init__(self,input_dim,relations=2,hidden=48,heads=4,mode="risk_gated_attention"):
        super().__init__()
        if mode not in MODES or hidden%heads:raise ValueError("Invalid model configuration")
        self.config=dict(input_dim=input_dim,relations=relations,hidden=hidden,heads=heads,mode=mode)
        self.mode=mode;self.heads=heads;self.hidden=hidden;self.relations=relations
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.keys=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(relations)])
        self.values=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(relations)])
        self.time_enc=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,hidden))
        self.gate_scale=nn.Parameter(torch.full((relations,),4.))
        self.gate_threshold=nn.Parameter(torch.full((relations,),-1.38629436))
        self.head=nn.Sequential(nn.Linear(hidden*(relations+1)+relations,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,1))

    def forward(self,x,neighbors,age_hours,mask,neighbor_risk,return_embedding=False):
        batch=x.shape[0];h=self.encoder(x);messages=[];coverage=[]
        for r in range(self.relations):
            m=mask[:,r];has=m.any(dim=1)
            if self.mode=="no_edge_mlp":
                message=torch.zeros_like(h);count=torch.zeros(batch,device=x.device)
            else:
                z=self.encoder(neighbors[:,r]);v=self.values[r](z)
                if self.mode=="relational_sage":
                    message=(v*m[:,:,None]).sum(1)/m.sum(1).clamp(min=1)[:,None]
                else:
                    k=self.keys[r](z)+self.time_enc(torch.log1p(age_hours[:,r].clamp(min=0))[:,:,None])
                    q=self.query(h).view(batch,self.heads,-1)
                    k=k.view(batch,-1,self.heads,self.hidden//self.heads)
                    logits=torch.einsum("bhd,bkhd->bhk",q,k)/(self.hidden//self.heads)**.5
                    weight=torch.softmax(logits.masked_fill(~m[:,None,:],-1e9),dim=-1)*m[:,None,:]
                    weight=weight/weight.sum(-1,keepdim=True).clamp(min=1e-12)
                    if self.mode=="risk_gated_attention":
                        gate=.1+.9*torch.sigmoid(F.softplus(self.gate_scale[r])*(neighbor_risk[:,r]-torch.sigmoid(self.gate_threshold[r])))
                        v=v*gate[:,:,None]
                    v=v.view(batch,-1,self.heads,self.hidden//self.heads)
                    message=torch.einsum("bhk,bkhd->bhd",weight,v).reshape(batch,self.hidden)
                message=message*has[:,None];count=m.float().mean(1)
            messages.append(message);coverage.append(count)
        embedding=torch.cat([h,*messages,torch.stack(coverage,dim=1)],dim=1)
        logits=self.head(embedding).squeeze(-1)
        return (logits,embedding) if return_embedding else logits

def tensors(x,graph,risk):
    return (torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(graph.neighbors,dtype=torch.long),
            torch.as_tensor(graph.age_hours,dtype=torch.float32),torch.as_tensor(risk,dtype=torch.float32))

def forward_rows(model,data,rows,embedding=False):
    x,adj,age,risk=data;ids=adj[rows];valid=ids>=0;safe=ids.clamp(min=0)
    return model(x[rows],x[safe],age[rows],valid,risk[safe],return_embedding=embedding)

def train(model,data,y,train_rows,epochs=12,batch_size=512,seed=42):
    """Fixed-epoch fit. No calibration, validation or test labels enter optimization."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed);target=torch.as_tensor(y,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
    for epoch in range(epochs):
        model.train();order=rng.permutation(train_rows);total=0
        for start in range(0,len(order),batch_size):
            rows=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            logits=forward_rows(model,data,rows)
            loss=F.binary_cross_entropy_with_logits(logits,target[rows]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(rows)
        history.append({"epoch":epoch+1,"train_bce":total/len(order)})
    model.eval();return history

def predict(model,data,rows,batch_size=1024,embedding=False):
    model.eval();scores=[];embeddings=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            selected=rows[start:start+batch_size]
            result=forward_rows(model,data,selected,embedding)
            if embedding:
                logits,z=result;embeddings.append(z.numpy())
            else:logits=result
            scores.append(torch.sigmoid(logits).numpy())
    score=np.concatenate(scores)
    return (score,np.concatenate(embeddings)) if embedding else score

def save_weights(model,path):
    path=Path(path);torch.save(model.state_dict(),path)
    manifest={"config":model.config,"torch":torch.__version__,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "format":"state_dict only; load with weights_only=True","production_approved":False}
    path.with_suffix(".json").write_text(json.dumps(manifest,indent=2))

def load_weights(path):
    path=Path(path);manifest=json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError("GNN checkpoint checksum mismatch")
    model=CausalRelationalGNN(**manifest['config'])
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();return model

