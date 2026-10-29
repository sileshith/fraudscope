"""User-supplied IEEE-CIS data adapter. No automatic download or redistribution."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .pipeline import run_experiment

def load_ieee(directory):
    root=Path(directory);tx=root/"train_transaction.csv";identity=root/"train_identity.csv"
    if not tx.exists(): raise FileNotFoundError("Place authorized train_transaction.csv in the supplied directory. See docs/IEEE_BENCHMARK.md")
    df=pd.read_csv(tx)
    required={"TransactionID","TransactionDT","TransactionAmt","isFraud"}
    if not required.issubset(df): raise ValueError(f"Missing columns: {required-set(df.columns)}")
    if df.TransactionID.isna().any() or df.TransactionID.duplicated().any(): raise ValueError("TransactionID must be unique and nonnull")
    if identity.exists():
        identities=pd.read_csv(identity)
        if "TransactionID" not in identities or identities.TransactionID.isna().any() or identities.TransactionID.duplicated().any():
            raise ValueError("Identity TransactionID must be unique and nonnull")
        if (set(df)&set(identities))!={"TransactionID"}: raise ValueError("Unexpected overlapping identity columns")
        df=df.merge(identities,on="TransactionID",how="left",validate="one_to_one")
    if not df.isFraud.isin([0,1]).all(): raise ValueError("isFraud must be binary")
    if not np.isfinite(df.TransactionDT).all() or (df.TransactionDT<0).any(): raise ValueError("Invalid TransactionDT")
    if not np.isfinite(df.TransactionAmt).all() or (df.TransactionAmt<=0).any(): raise ValueError("Invalid TransactionAmt")
    df=df.sort_values(["TransactionDT","TransactionID"],kind="stable").reset_index(drop=True)
    # Allowlist removes target, ID, arbitrary added columns and all post-outcome metadata.
    fixed={"TransactionAmt","ProductCD","card1","card2","card3","card4","card5","card6",
           "addr1","addr2","dist1","dist2","P_emaildomain","R_emaildomain","DeviceType","DeviceInfo"}
    allowed=[c for c in df if c in fixed or re.fullmatch(r"(?:C|D|M|V)\d+|id_\d+",c)]
    x=df[allowed].copy()
    categorical=[c for c in allowed if x[c].dtype==object or c.startswith("card") or c.startswith("addr") or c.startswith("M") or (c.startswith("id_") and int(c[3:])>=12)]
    numeric=[c for c in allowed if c not in categorical]
    for c in categorical: x[c]=x[c].fillna("unknown").astype(str)
    for c in numeric: x[c]=pd.to_numeric(x[c],errors="raise").replace([np.inf,-np.inf],np.nan)
    return df,x,numeric,categorical

def run_ieee(directory,out="reports/ieee",seed=42,lag_days=2):
    df,x,numeric,categorical=load_ieee(directory)
    return run_experiment(df,x,numeric,categorical,out,seed,"ieee_cis",time_col="TransactionDT",label_col="isFraud",amount_col="TransactionAmt",lag_days=lag_days)

