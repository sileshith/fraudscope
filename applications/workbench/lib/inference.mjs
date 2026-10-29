import models from '../model-data/models.json' with {type:'json'};
export const modelVersion=models.version;
export const activePolicy=models.active.policy;
export const graphModes=Object.keys(models.gnn);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sigmoid=x=>1/(1+Math.exp(-x));
const dense=(x,w,b)=>w.map((row,i)=>dot(row,x)+(b?.[i]??0));
function erf(x){const sign=x<0?-1:1;x=Math.abs(x);const t=1/(1+.3275911*x);return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));}
const gelu=x=>.5*x*(1+erf(x/Math.sqrt(2)));
export function encode(f,p=models.graph_pre){const n=p.numeric.map((key,i)=>(f[key]-p.mean[i])/p.scale[i]);return [...n,...p.categorical.flatMap((key,i)=>p.categories[i].map(v=>Number(f[key]===v)))];}
export function calibrate(p,c){if(p<=c.x[0])return c.y[0];const last=c.x.length-1;if(p>=c.x[last])return c.y[last];let lo=0,hi=last;while(hi-lo>1){const mid=(lo+hi)>>1;if(c.x[mid]<=p)lo=mid;else hi=mid;}return c.y[lo]+(c.y[hi]-c.y[lo])*(p-c.x[lo])/(c.x[hi]-c.x[lo]);}
export function linearScore(f,teacher=false){const m=teacher?models.teacher:models.active;const raw=sigmoid(dot(encode(f,m.pre),m.coef)+m.intercept);return {raw,probability:teacher?raw:calibrate(raw,m.calibration)};}
export function graphScore(encoded,neighbors,mode='risk_gated_attention'){
 const m=models.gnn[mode];if(!m)throw Error('Unknown graph model');const w=m.weights;const H=m.config.hidden;const D=H/m.config.heads;
 const linear=(x,p)=>dense(x,w[p+'.weight'],w[p+'.bias']);
 const enc=x=>{const a=linear(x,'encoder.0'),mean=a.reduce((s,v)=>s+v,0)/H;const variance=a.reduce((s,v)=>s+(v-mean)**2,0)/H;return a.map((v,i)=>gelu((v-mean)/Math.sqrt(variance+1e-5)*w['encoder.1.weight'][i]+w['encoder.1.bias'][i]));};
 const h=enc(encoded),q=linear(h,'query'),messages=[],coverage=[];
 for(let r=0;r<2;r++){
  const ns=neighbors[r].slice(0,8);coverage.push(mode==='no_edge_mlp'?0:ns.length/8);let message=Array(H).fill(0);
  if(mode!=='no_edge_mlp'&&ns.length){const z=ns.map(n=>enc(n.encoded)),vs=z.map(v=>linear(v,'values.'+r));
   if(mode==='relational_sage'){message=message.map((_,i)=>vs.reduce((s,v)=>s+v[i],0)/ns.length);}
   else {const ks=z.map((v,i)=>{const time=linear(linear([Math.log1p(Math.max(0,ns[i].age_hours))],'time_enc.0').map(Math.tanh),'time_enc.2');return linear(v,'keys.'+r).map((x,j)=>x+time[j]);});
    const gates=ns.map(n=>mode==='risk_gated_attention'?.1+.9*sigmoid(Math.log1p(Math.exp(w.gate_scale[r]))*(n.risk-sigmoid(w.gate_threshold[r]))):1);
    for(let head=0;head<m.config.heads;head++){const start=head*D;const logits=ks.map(k=>dot(q.slice(start,start+D),k.slice(start,start+D))/Math.sqrt(D));const max=Math.max(...logits),ex=logits.map(v=>Math.exp(v-max)),den=ex.reduce((s,v)=>s+v,0);for(let d=0;d<D;d++)message[start+d]=vs.reduce((s,v,i)=>s+(ex[i]/den)*v[start+d]*gates[i],0);}
   }
  }messages.push(...message);
 }
 const embedding=[...h,...messages,...coverage];const raw=sigmoid(linear(linear(embedding,'head.0').map(gelu),'head.3')[0]);return {raw,probability:calibrate(raw,m.calibration),threshold:m.policy.threshold,mode};
}
export function extractFeatures(e,history){const t=e.event_time,DAY=86400;const ah=history.filter(r=>r.account_id===e.account_id&&r.event_time>t-7*DAY&&r.event_time<t),dh=history.filter(r=>r.device_id===e.device_id&&r.event_time>t-DAY&&r.event_time<t);const mean=ah.length?ah.reduce((s,r)=>s+r.amount,0)/ah.length:e.amount;return {log_amount:Math.log1p(e.amount),account_age_days:e.account_age_days,email_age_days:e.email_age_days,geo_distance_km:e.geo_distance_km,failed_logins_24h:e.failed_logins_24h,cross_border:e.cross_border,channel:e.channel,product:e.product,device_type:e.device_type,hour_sin:Math.sin(t%DAY/DAY*2*Math.PI),hour_cos:Math.cos(t%DAY/DAY*2*Math.PI),account_txn_1h:ah.filter(r=>r.event_time>t-3600).length,account_txn_24h:ah.filter(r=>r.event_time>t-DAY).length,amount_ratio_7d:Math.min(e.amount/Math.max(mean,.01),1000),device_accounts_24h:new Set(dh.map(r=>r.account_id)).size,new_device_7d:Number(!ah.some(r=>r.device_id===e.device_id)),account_txn_7d:ah.length};}
export function neighborsFor(e,history){return ['account_id','device_id'].map((key,r)=>history.filter(v=>v[key]===e[key]&&v.event_time<e.event_time&&v.event_time>e.event_time-(r===0?7:1)*86400).sort((a,b)=>b.event_time-a.event_time||b.seq-a.seq).slice(0,8).map(v=>({id:v.transaction_id,encoded:JSON.parse(v.encoded_json),risk:v.teacher_risk,age_hours:(e.event_time-v.event_time)/3600})));}
export function evidence(f){const a=[];if(f.account_txn_1h>=3)a.push(`${f.account_txn_1h} earlier account transactions in one hour`);if(f.device_accounts_24h>=2)a.push(`Device shared by ${f.device_accounts_24h} earlier accounts in 24 hours`);if(f.amount_ratio_7d>=3)a.push(`Amount is ${f.amount_ratio_7d.toFixed(1)} times the prior seven-day mean`);if(f.new_device_7d)a.push('Device not seen on this account in the prior seven days');if(f.failed_logins_24h>0)a.push(`${f.failed_logins_24h} failed logins reported in 24 hours`);if(f.cross_border)a.push('Cross-border transaction');if(f.geo_distance_km>1000)a.push(`${Math.round(f.geo_distance_km).toLocaleString('en-US')} km reported geographic distance`);return a.length?a:['No listed rule signal triggered. The model still uses all validated features.'];}
