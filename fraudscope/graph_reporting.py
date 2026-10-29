import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def graph_report(out,result,stress):
    out=Path(out);models=result['models'];names=[r['model'].replace('_',' ') for r in models]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,key,title in [(axes[0],'average_precision','Final holdout average precision'),(axes[1],'recall_at_5pct','Fraud recall within top 5%')]:
        ax.barh(names[::-1],[r['test'][key] for r in models][::-1],color=['#087f8c' if r['model'] in ['temporal_attention','risk_gated_attention'] else '#8aa7ad' for r in models][::-1])
        ax.set_title(title);ax.set_xlim(0,1)
    for kind,color in [('delete','#087f8c'),('rewire','#b67924')]:
        mean=[];std=[]
        for q in [.1,.3]:
            v=[r['threshold_flip_rate'] for r in stress if r['kind']==kind and r['requested_fraction']==q]
            mean.append(np.mean(v));std.append(np.std(v,ddof=1))
        axes[2].errorbar([.1,.3],mean,yerr=std,label=kind,color=color,marker='o',capsize=4)
    axes[2].set(xlabel='Fraction of incoming test edges perturbed',ylabel='Frozen-policy decision flip rate',title='Structural sensitivity (3 perturbation draws)');axes[2].legend()
    fig.suptitle(f"FraudScope GNN research extension | synthetic seed {result['seed']}",fontsize=15)
    fig.savefig(out/'gnn_comparison.png',dpi=160);plt.close(fig)
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['model'],f"{r['test']['average_precision']:.4f}",f"{r['test']['roc_auc']:.4f}",f"{100*r['test']['recall_at_5pct']:.1f}%",f"{100*r['test'].get('review_rate',0):.1f}%" if 'review_rate' in r['test'] else 'Not selected'])+'</tr>' for r in models)
    stressrows=''.join(f"<tr><td>{r['kind']}</td><td>{r['requested_fraction']:.0%}</td><td>{r['replicate']}</td><td>{r['mean_absolute_score_drift']:.4f}</td><td>{r['threshold_flip_rate']:.2%}</td><td>{r['average_precision']:.4f}</td></tr>" for r in stress)
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FraudScope GNN evidence</title><style>
body{{font-family:system-ui,sans-serif;background:#f2f6f5;color:#14333d;max-width:1100px;margin:0 auto;padding:32px}}h1,h2{{font-family:Georgia,serif;font-weight:500}}h1{{font-size:42px}}p,li{{line-height:1.65}}.card{{background:white;padding:24px;margin:20px 0;border:1px solid #dce6e6;border-radius:10px}}.notice{{border-left:4px solid #b67924;padding:16px;background:#fff7e9}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;padding:12px 8px;border-bottom:1px solid #dce6e6}}.scroll{{overflow:auto}}a{{color:#087f8c}}.tag{{color:#087f8c;letter-spacing:2px;font-size:12px}}code{{font-size:12px}}@media(max-width:600px){{body{{padding:16px}}h1{{font-size:30px}}.card{{padding:16px}}}}</style>
<p class="tag">FRAUDSCOPE / GNN RESEARCH EXTENSION</p><h1>Learn from earlier connections.</h1>
<p>Temporal relational message passing, risk gates and structural sensitivity, built for a researcher who wants to connect fraud analytics with GNN stability.</p>
<div class="notice">Synthetic pilot only. Real IEEE-CIS results remain pending. This is an original reference implementation inspired by research, not a reproduction of HGT, TGN or RGH-GNN. The scoring API still serves its validated tabular model.</div>
<section class="card"><h2>Four neural candidates added</h2><ul><li>No-edge MLP: isolates the value of neural nonlinearities.</li><li>Relational GraphSAGE-style mean aggregation: establishes a simple graph baseline.</li><li>Temporal relational attention: four attention heads, typed neighbor paths and relative event time.</li><li>Risk-gated temporal attention: adds earlier, out-of-fold teacher risk to the message values.</li></ul>
<p>Graph: {result['directed_edges']:,} directed historical edges, at most {result['config']['neighbors_per_relation']} earlier neighbors per relation. Relations: {html.escape(', '.join(result['graph_relations']))}. Equal-time and future edges are forbidden. Labels never become node features or edge signs.</p></section>
<section class="card"><h2>Same-split model comparison</h2><p>Seed {result['seed']}. Choice by validation cost: <b>{result['overall_candidate_winner']}</b>. Best graph candidate by that rule: <b>{result['graph_candidate_winner']}</b>. These candidates were fixed before their test evaluation. This extension follows an already inspected synthetic benchmark and remains exploratory.</p><div class="scroll"><table><tr><th>Model</th><th>Test AP</th><th>ROC-AUC</th><th>Recall at 5%</th><th>Policy review rate</th></tr>{rows}</table></div></section>
<section class="card"><h2>Bridge to structural robustness</h2><p>Weights, calibration and threshold are frozen. Edges entering test transactions are deleted or rewired to strictly earlier nodes. Rewiring preserves target indegree where possible but breaks real entity membership. It is a controlled stress test, not a claim about realistic attacker behavior.</p><div class="scroll"><table><tr><th>Change</th><th>Budget</th><th>Draw</th><th>Mean score drift</th><th>Decision flips</th><th>AP</th></tr>{stressrows}</table></div><p>JSON evidence also records raw adjacency displacement and embedding drift. A reversal that introduces future information is rejected. Signed or magnetic GNNs require a defensible edge-sign meaning and are not trained here.</p></section>
<section class="card"><h2>Research basis and limits</h2><p><a href="https://arxiv.org/abs/2205.13084">BRIGHT (2022)</a> motivates history-only information flow. <a href="https://arxiv.org/abs/2003.01332">HGT (2020)</a> motivates relation and time-aware attention. <a href="https://arxiv.org/abs/2610.27107">RGH-GNN (October 2026 preprint)</a> motivates risk-guided messages. The newest paper's stratified evaluation is not comparable to this chronological protocol.</p><p>No persistent TGN memory or full heterogeneous entity-node states are claimed. Teacher scoring uses past-only logistic fits rather than the paper's LightGBM pipeline. The three simulation seeds and limited perturbations are pilot evidence, not a stability theorem or deployment validation.</p></section></html>'''
    (out/'gnn_report.html').write_text(report)
