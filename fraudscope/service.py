"""Single-node reference scoring API. Strict event ordering is part of its contract."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import hashlib
import hmac
import json
import os
import sqlite3
import time
import pandas as pd
from fastapi import FastAPI,Depends,HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel,ConfigDict,Field
from .contracts import Transaction,FEATURES,FEATURE_VERSION
from .features import extract,DAY
from .models import load_bundle
from .explain import evidence_brief

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def state_hash(row):
    return hashlib.sha256(canonical({k:row[k] for k in ["event_time","account_id","device_id","amount"]}).encode()).hexdigest()

def connect(path):
    con=sqlite3.connect(path,timeout=10);con.row_factory=sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con

def init_db(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,transaction_id TEXT UNIQUE NOT NULL,
          payload_hash TEXT NOT NULL,event_time REAL NOT NULL,account_id TEXT NOT NULL,
          device_id TEXT NOT NULL,amount REAL NOT NULL,features TEXT NOT NULL,
          response TEXT NOT NULL,audit_record TEXT NOT NULL,previous_hash TEXT NOT NULL,audit_hash TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS account_time ON events(account_id,event_time);
        CREATE INDEX IF NOT EXISTS device_time ON events(device_id,event_time);
        CREATE INDEX IF NOT EXISTS global_time ON events(event_time);
        CREATE TABLE IF NOT EXISTS feedback (
          transaction_id TEXT PRIMARY KEY REFERENCES events(transaction_id),label INTEGER NOT NULL CHECK(label IN (0,1)),
          available_time REAL NOT NULL,source TEXT NOT NULL);
        """)

def verify_audit(path):
    prev="0"*64;count=0
    with connect(path) as db:
        for row in db.execute("SELECT * FROM events ORDER BY seq"):
            digest=hashlib.sha256((prev+row["audit_record"]).encode()).hexdigest()
            record=json.loads(row["audit_record"])
            if row["previous_hash"]!=prev or digest!=row["audit_hash"] or record["transaction_id"]!=row["transaction_id"] or record["payload_hash"]!=row["payload_hash"] or record["response_hash"]!=hashlib.sha256(row["response"].encode()).hexdigest() or record["features_hash"]!=hashlib.sha256(row["features"].encode()).hexdigest() or record["state_hash"]!=state_hash(row):
                return {"valid":False,"checked":count,"failed_sequence":row["seq"]}
            prev=digest;count+=1
    return {"valid":True,"checked":count,"head":prev,"note":"Tamper evidence only. Anchor the head externally to detect wholesale rewrite or truncation."}

class Feedback(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    transaction_id:str=Field(min_length=1,max_length=64)
    label:Literal[0,1]
    available_time:float=Field(ge=0,le=4102444800)
    source:Literal["confirmed_chargeback","confirmed_legitimate"]

def create_app(model_path=None,db_path=None,api_key=None):
    model_path=Path(model_path or os.getenv("FRAUDSCOPE_MODEL","reports/demo/seed_42/model.joblib"))
    db_path=str(db_path or os.getenv("FRAUDSCOPE_DB","runtime/fraudscope.sqlite"))
    key=api_key or os.getenv("FRAUDSCOPE_API_KEY","")
    header=APIKeyHeader(name="X-API-Key",auto_error=False)
    def authorized(value=Depends(header)):
        if not value or not hmac.compare_digest(value,key): raise HTTPException(401,"Invalid API key")
    @asynccontextmanager
    async def lifespan(app):
        if len(key)<24: raise RuntimeError("Set FRAUDSCOPE_API_KEY to a random secret of at least 24 characters")
        bundle=load_bundle(model_path)
        if bundle.dataset!="synthetic" or bundle.feature_version!=FEATURE_VERSION:
            raise RuntimeError("This raw-event API requires the synthetic-schema model. Build a separate contract for IEEE/real data.")
        init_db(db_path);app.state.bundle=bundle
        yield
    app=FastAPI(title="FraudScope scoring reference",version="1.0.0",lifespan=lifespan)

    @app.get("/healthz")
    def health():return {"status":"ok"}

    @app.get("/readyz",dependencies=[Depends(authorized)])
    def ready():
        with connect(db_path) as db: db.execute("SELECT 1 FROM events LIMIT 1").fetchone()
        return {"status":"ready","model_version":app.state.bundle.model_version,"production_approved":False}

    @app.post("/v1/score",dependencies=[Depends(authorized)])
    def score(event:Transaction):
        started=time.perf_counter();e=event.model_dump();payload_hash=hashlib.sha256(canonical(e).encode()).hexdigest()
        db=connect(db_path)
        try:
            db.execute("BEGIN IMMEDIATE")
            old=db.execute("SELECT payload_hash,response FROM events WHERE transaction_id=?",(e["transaction_id"],)).fetchone()
            if old:
                if old["payload_hash"]!=payload_hash:raise HTTPException(409,"Transaction ID already exists with different content")
                return json.loads(old["response"])
            last=db.execute("SELECT MAX(event_time) FROM events").fetchone()[0]
            if last is not None and e["event_time"]<last:
                raise HTTPException(409,"Out-of-order event. Send to the late-event reconciliation path; no state was changed.")
            t=e["event_time"]
            ah=db.execute("SELECT event_time,amount,device_id FROM events WHERE account_id=? AND event_time>? AND event_time<? ORDER BY event_time",(e["account_id"],t-7*DAY,t)).fetchall()
            dh=db.execute("SELECT event_time,account_id FROM events WHERE device_id=? AND event_time>? AND event_time<? ORDER BY event_time",(e["device_id"],t-DAY,t)).fetchall()
            f=extract(e,ah,dh);b=app.state.bundle
            probability=float(b.predict(pd.DataFrame([f])[FEATURES])[0])
            decision="review" if probability>=b.policy["threshold"] else "pass"
            response={"transaction_id":e["transaction_id"],"fraud_probability":probability,"decision":decision,
                      "model_version":b.model_version,"feature_version":b.feature_version,"threshold":b.policy["threshold"],
                      "processing_ms":round((time.perf_counter()-started)*1000,3),
                      "brief":evidence_brief(f,probability,decision,b.model_version),"automatic_block":False}
            response_text=canonical(response);feature_text=canonical(f)
            previous=db.execute("SELECT audit_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev=previous[0] if previous else "0"*64
            audit=canonical({"transaction_id":e["transaction_id"],"payload_hash":payload_hash,"state_hash":state_hash(e),
                            "response_hash":hashlib.sha256(response_text.encode()).hexdigest(),
                            "features_hash":hashlib.sha256(feature_text.encode()).hexdigest()})
            digest=hashlib.sha256((prev+audit).encode()).hexdigest()
            db.execute("INSERT INTO events(transaction_id,payload_hash,event_time,account_id,device_id,amount,features,response,audit_record,previous_hash,audit_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (e["transaction_id"],payload_hash,t,e["account_id"],e["device_id"],e["amount"],feature_text,response_text,audit,prev,digest))
            db.commit();return response
        except sqlite3.OperationalError as exc:
            raise HTTPException(503,"State store unavailable; retry with the same transaction ID") from exc
        finally:
            if db.in_transaction:db.rollback()
            db.close()

    @app.post("/v1/feedback",dependencies=[Depends(authorized)])
    def feedback(item:Feedback):
        if item.label!=int(item.source=="confirmed_chargeback"):raise HTTPException(422,"Label and source disagree")
        with connect(db_path) as db:
            row=db.execute("SELECT event_time FROM events WHERE transaction_id=?",(item.transaction_id,)).fetchone()
            if row is None:raise HTTPException(404,"Transaction not found")
            if item.available_time<row["event_time"]:raise HTTPException(422,"Outcome cannot precede the event")
            old=db.execute("SELECT label,available_time,source FROM feedback WHERE transaction_id=?",(item.transaction_id,)).fetchone()
            values=(item.label,item.available_time,item.source)
            if old and tuple(old)!=values:raise HTTPException(409,"Outcome already exists. Reconcile corrections through a governed workflow.")
            db.execute("INSERT OR IGNORE INTO feedback VALUES(?,?,?,?)",(item.transaction_id,*values))
        return {"stored":True,"automatically_retrained":False}

    @app.get("/v1/operations",dependencies=[Depends(authorized)])
    def operations():
        with connect(db_path) as db:
            count=db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            labels=db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            recent=[json.loads(r[0]) for r in db.execute("SELECT response FROM events ORDER BY seq DESC LIMIT 1000")]
        return {"events":count,"feedback_count":labels,"recent_window_n":len(recent),
                "recent_review_rate":sum(r["decision"]=="review" for r in recent)/max(len(recent),1),
                "capacity_target":app.state.bundle.policy["capacity_target"],
                "latency_ms_p95":float(pd.Series([r["processing_ms"] for r in recent]).quantile(.95)) if recent else None}

    @app.get("/v1/audit",dependencies=[Depends(authorized)])
    def audit():return verify_audit(db_path)
    return app

