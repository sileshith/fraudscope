import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(prog="fraudscope");subs=p.add_subparsers(dest="command",required=True)
    demo=subs.add_parser("demo");demo.add_argument("--rows",type=int,default=60000);demo.add_argument("--seeds",type=int,nargs="+",default=[42,73,101]);demo.add_argument("--out",default="reports/demo")
    ieee=subs.add_parser("benchmark");ieee.add_argument("--data",required=True);ieee.add_argument("--out",default="reports/ieee");ieee.add_argument("--seed",type=int,default=42);ieee.add_argument("--label-lag-days",type=float,default=2)
    monitor=subs.add_parser("monitor");monitor.add_argument("--reference",required=True);monitor.add_argument("--current",required=True);monitor.add_argument("--out",default="reports/drift_current.json")
    audit=subs.add_parser("audit");audit.add_argument("--db",required=True)
    feedback=subs.add_parser("evaluate-feedback");feedback.add_argument("--db",required=True);feedback.add_argument("--as-of",type=float,required=True);feedback.add_argument("--lag-days",type=float,default=2)
    args=p.parse_args()
    if args.command=="demo":
        from .pipeline import run_demo
        print(json.dumps(run_demo(args.out,args.rows,args.seeds),indent=2))
    elif args.command=="benchmark":
        if args.label_lag_days<0:p.error("label lag cannot be negative")
        from .benchmark import run_ieee
        print(json.dumps(run_ieee(args.data,args.out,args.seed,args.label_lag_days)["test"],indent=2))
    elif args.command=="monitor":
        import pandas as pd
        from .monitor import drift_report
        from .pipeline import write_json
        result=drift_report(json.loads(Path(args.reference).read_text()),pd.read_csv(args.current));write_json(args.out,result);print(json.dumps(result,indent=2))
    elif args.command=="evaluate-feedback":
        from .feedback import evaluate_feedback
        print(json.dumps(evaluate_feedback(args.db,args.as_of,args.lag_days),indent=2))
    else:
        from .service import verify_audit
        result=verify_audit(args.db);print(json.dumps(result,indent=2));raise SystemExit(0 if result["valid"] else 1)

if __name__=="__main__":main()
