import argparse
import json
from fraudscope.graph_experiment import run_graph_demo,run_graph_experiment

p=argparse.ArgumentParser(description='Optional GNN research experiment; no API model promotion')
p.add_argument('--rows',type=int,default=60000);p.add_argument('--seeds',nargs='+',type=int,default=[42,73,101])
p.add_argument('--epochs',type=int,default=12);p.add_argument('--neighbors',type=int,default=8)
p.add_argument('--out',default='reports/gnn');p.add_argument('--baseline-root',default='reports/demo')
p.add_argument('--ieee-data',default=None,help='Authorized IEEE-CIS CSV folder. Trains a new schema-specific GNN.')
p.add_argument('--label-lag-days',type=float,default=2)
args=p.parse_args()
if args.ieee_data:
    from fraudscope.benchmark import load_ieee
    df,x,num,cat=load_ieee(args.ieee_data)
    if args.label_lag_days<0:p.error('Label lag must be nonnegative')
    result=run_graph_experiment(df,x,num,cat,args.out,args.seeds[0],args.epochs,args.neighbors,dataset='ieee_cis',time_col='TransactionDT',label_col='isFraud',amount_col='TransactionAmt',lag_days=args.label_lag_days)
    print(json.dumps({'status':'local_ieee_benchmark','winner':result['graph_candidate_winner']},indent=2))
else:print(json.dumps(run_graph_demo(args.out,args.rows,args.seeds,args.epochs,args.neighbors,args.baseline_root),indent=2))
