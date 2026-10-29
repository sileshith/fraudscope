"""Local, self-contained portfolio dashboard and shareable static figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

def make_report(out,result,prediction,drift):
    out=Path(out);p=prediction.score.to_numpy();y=prediction.label.to_numpy();amount=prediction.amount.to_numpy()
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    plt.rcParams.update({"font.size":11})
    pr,re,_=precision_recall_curve(y,p)
    ax[0,0].plot(re,pr,color="#087f8c",lw=2);ax[0,0].axhline(y.mean(),color="#64748b",ls="--",label="Fraud prevalence")
    ax[0,0].set(xlabel="Recall",ylabel="Precision",title="Final holdout: precision-recall");ax[0,0].legend()
    frac,mean=calibration_curve(y,p,n_bins=8,strategy="quantile")
    ax[0,1].plot([0,1],[0,1],ls="--",color="#64748b");ax[0,1].plot(mean,frac,"o-",color="#087f8c")
    ax[0,1].set(xlabel="Mean predicted probability",ylabel="Observed fraud rate",title="Calibration (quantile bins)")
    order=np.argsort(-p,kind="stable");budget=np.arange(1,len(y)+1)/len(y)
    ax[1,0].plot(budget,np.cumsum(y[order])/y.sum(),label="Fraud count",color="#087f8c")
    ax[1,0].plot(budget,np.cumsum((y*amount)[order])/(y*amount).sum(),label="Fraud value",color="#dd9730")
    ax[1,0].plot([0,1],[0,1],ls="--",color="#64748b");ax[1,0].set(xlim=(0,.3),xlabel="Fraction reviewed",ylabel="Fraction captured",title="Retrospective ranked review");ax[1,0].legend()
    top=drift["features"][:7]
    ax[1,1].barh([r["feature"] for r in top][::-1],[r["psi"] for r in top][::-1],color="#087f8c")
    ax[1,1].axvline(.2,color="#dd9730",ls="--");ax[1,1].set(xlabel="Population stability index",title="Train-to-test input drift")
    fig.suptitle(f"FraudScope | {result['dataset']} | seed {result['seed']} | {result['winner']}",fontsize=15)
    fig.savefig(out/"evaluation.png",dpi=170);plt.close(fig)
    cases=json.loads((out/"cases.json").read_text()) if (out/"cases.json").exists() else []
    packed={"metrics":result,"drift":drift,"cases":cases,
            "predictions":np.round(prediction[["score","label","amount"]].to_numpy(),6).tolist()}
    data=json.dumps(packed,allow_nan=False).replace("</","<\\/")
    template=Path(__file__).with_name("dashboard.html").read_text()
    (out/"portfolio.html").write_text(template.replace("__REPORT_DATA__",data))

