"""Scenario simulator, not an estimate of the real IEEE-CIS distribution."""
import numpy as np
import pandas as pd

def generate(n: int = 60000, seed: int = 42, days: int = 120) -> pd.DataFrame:
    if n < 1000 or days < 20:
        raise ValueError("Use at least 1000 transactions and 20 days")
    rng = np.random.default_rng(seed)
    accounts = max(250, n // 15)
    times = np.sort(rng.uniform(0, days * 86400, n))
    users = rng.integers(0, accounts, n)
    intent = rng.random(n) < .044
    # Repeated accounts and shared devices create weak network/velocity signals.
    for i in range(1, n):
        if intent[i-1] and rng.random() < .48:
            intent[i] = True
            users[i] = users[i-1]
    late = times > .75 * days * 86400
    scenario = rng.choice(["account_takeover", "card_testing", "social_engineering"], n, p=[.55,.25,.20])
    scenario[late & intent & (rng.random(n)<.45)] = "social_engineering"
    baseline_spend = rng.lognormal(3.65,.55,accounts)
    signup_age = rng.exponential(240,accounts)
    amount = baseline_spend[users] * rng.lognormal(0,.85,n)
    takeover = intent & (scenario == "account_takeover")
    testing = intent & (scenario == "card_testing")
    social = intent & (scenario == "social_engineering")
    amount[takeover | social] *= 2.0
    amount[testing] *= .12
    amount = np.clip(amount, .5, 8000).round(2)
    fresh = rng.random(n) < np.where(takeover | testing,.67,.12)
    devices = np.array([f"dev_{u}" for u in users], dtype=object)
    for i in np.flatnonzero(fresh):
        devices[i] = f"shared_{rng.integers(0,80)}" if intent[i] else f"new_{rng.integers(0,accounts*2)}"
    distance = rng.exponential(28,n)
    far = rng.random(n) < np.where(takeover | testing, .65, .09)
    distance[far] = rng.uniform(200,9000,far.sum())
    failures = rng.poisson(np.where(takeover,1.7,.12))
    email_age = signup_age[users] * rng.uniform(.3,1.3,n)
    email_age[testing] *= .05
    # Outcomes are stochastic: suspicious does not mean fraud, and some fraud is quiet.
    y = (rng.random(n) < np.where(intent,.72,.0025)).astype(int)
    df = pd.DataFrame({
        "transaction_id": [f"s{seed}_{i:08}" for i in range(n)], "event_time": times,
        "account_id": [f"acct_{u}" for u in users], "device_id": devices, "amount": amount,
        "account_age_days": signup_age[users]+times/86400, "email_age_days": email_age,
        "geo_distance_km": distance, "failed_logins_24h": failures,
        "cross_border": (rng.random(n) < np.where(far,.64,.06)).astype(int),
        "channel": np.where(rng.random(n)<np.where(late,.72,.48),"app","web"),
        "product": rng.choice(["retail","digital","travel","subscription"],n,p=[.52,.25,.13,.10]),
        "device_type": rng.choice(["desktop","mobile","tablet"],n,p=[.35,.57,.08]),
        "is_fraud": y, "label_available_time": times+2*86400,
        "scenario_audit_only": np.where(intent,scenario,"ordinary"),
    })
    return df

