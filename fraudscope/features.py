"""One point-in-time feature definition shared by training and online serving."""
from collections import defaultdict, deque
import math
import numpy as np
import pandas as pd
from .contracts import FEATURES

DAY = 86400

def extract(event: dict, account_history: list, device_history: list) -> dict:
    """History tuples are (time, amount, device) and (time, account).

    Strictly earlier events only. Same-time events cannot observe each other.
    No labels are read. Rolling windows are (t-window, t).
    """
    t = event["event_time"]
    ah = [r for r in account_history if t-7*DAY < r[0] < t]
    dh = [r for r in device_history if t-DAY < r[0] < t]
    mean = np.mean([r[1] for r in ah]) if ah else event["amount"]
    hour = (t % DAY)/DAY*2*math.pi
    return {
        "log_amount": math.log1p(event["amount"]),
        **{c:event[c] for c in ["account_age_days","email_age_days","geo_distance_km",
                               "failed_logins_24h","cross_border","channel","product","device_type"]},
        "hour_sin": math.sin(hour), "hour_cos": math.cos(hour),
        "account_txn_1h": sum(r[0]>t-3600 for r in ah),
        "account_txn_24h": sum(r[0]>t-DAY for r in ah),
        "amount_ratio_7d": min(event["amount"]/max(float(mean),.01),1000),
        "device_accounts_24h": len({r[1] for r in dh}),
        "new_device_7d": int(event["device_id"] not in {r[2] for r in ah}),
        "account_txn_7d": len(ah),
    }

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if not df.event_time.is_monotonic_increasing:
        raise ValueError("Events must be sorted chronologically")
    if df.transaction_id.duplicated().any():
        raise ValueError("Duplicate transaction IDs")
    ah, dh = defaultdict(deque), defaultdict(deque)
    rows=[]
    for e in df.to_dict("records"):
        t=e["event_time"]; a=ah[e["account_id"]]; d=dh[e["device_id"]]
        while a and a[0][0]<=t-7*DAY: a.popleft()
        while d and d[0][0]<=t-DAY: d.popleft()
        rows.append(extract(e,a,d))
        a.append((t,e["amount"],e["device_id"]))
        d.append((t,e["account_id"]))
    return pd.DataFrame(rows,index=df.index)[FEATURES]

