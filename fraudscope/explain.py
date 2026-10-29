"""Evidence briefs are deterministic and do not invent fraud narratives."""
import numpy as np
import pandas as pd

POLICIES={
    "P01":"A score prioritizes review. It does not prove fraud.",
    "P02":"Verify the transaction through an approved independent channel.",
    "P03":"Do not block, contact, or change an account from this service.",
    "P04":"Review limits and recovery rates are planning assumptions.",
}

def evidence_brief(features,score,decision,model_version):
    evidence=[]
    conditions=[("account_txn_1h",lambda v:v>=2,"Multiple earlier transactions within one hour"),
                ("device_accounts_24h",lambda v:v>=2,"Device observed on multiple accounts within 24 hours"),
                ("amount_ratio_7d",lambda v:v>=3,"Amount is above the account's seven-day mean"),
                ("new_device_7d",lambda v:v==1,"Device not observed for this account in the seven-day window"),
                ("failed_logins_24h",lambda v:v>=2,"Several recent failed sign-ins"),
                ("geo_distance_km",lambda v:v>=500,"Large distance from the supplied reference location")]
    for name,predicate,text in conditions:
        if name in features and predicate(features[name]): evidence.append({"feature":name,"observed":features[name],"statement":text})
    return {"mode":"deterministic_evidence_brief","model_version":model_version,
            "score":round(float(score),6),"decision":decision,"evidence":evidence,
            "caveat":"Evidence rules are observations, not model attributions or proof of fraud.",
            "policy_references":POLICIES,"next_action":"Analyst verification" if decision=="review" else "Continue standard controls",
            "external_model_calls":0}

def reference_sensitivity(bundle,row,limit=6):
    """One-feature replacement diagnostic. NOT SHAP, causal effects or additive explanations."""
    x=pd.DataFrame([row])[bundle.feature_names]; original=float(bundle.predict(x)[0]); rows=[]
    for c in bundle.feature_names:
        copy=x.copy(); copy[c]=bundle.reference[c]
        rows.append({"feature":c,"score_change":original-float(bundle.predict(copy)[0]),"reference":bundle.reference[c]})
    return sorted(rows,key=lambda r:-abs(r["score_change"]))[:limit]

