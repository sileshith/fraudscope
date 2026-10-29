import numpy as np
import pandas as pd

def temporal_split(df: pd.DataFrame, time_col="event_time", label_time_col="label_available_time", lag_days=2):
    """55/15/10/20% of elapsed time, not shuffled row fractions.

    Train labels must mature before calibration starts, calibration labels before
    validation, and validation labels before the final test. End-of-period labels
    are purged. Synthetic test labels are assumed available at retrospective readout.
    """
    t=df[time_col].to_numpy(float)
    if len(t)<100 or not np.isfinite(t).all() or np.any(np.diff(t)<0) or t[-1]<=t[0]:
        raise ValueError("Need at least 100 finite, sorted, nonconstant event times")
    b=t[0]+(t[-1]-t[0])*np.array([.55,.70,.80])
    available=df[label_time_col].to_numpy(float) if label_time_col in df else t+lag_days*86400
    if not np.isfinite(available).all() or np.any(available<t):
        raise ValueError("Label availability must be finite and cannot precede event time")
    masks={"train":(t<b[0])&(available<b[0]),
           "calibration":(t>=b[0])&(t<b[1])&(available<b[1]),
           "validation":(t>=b[1])&(t<b[2])&(available<b[2]), "test":t>=b[2]}
    idx={k:np.flatnonzero(v) for k,v in masks.items()}
    if min(map(len,idx.values()))<20:
        raise ValueError("Too few rows after maturity purge; increase data span")
    meta={"time_boundaries":b.tolist(),"label_lag_days":lag_days,
          "counts":{k:len(v) for k,v in idx.items()},
          "purged_rows":int(len(t)-sum(map(len,idx.values())))}
    return idx,meta

