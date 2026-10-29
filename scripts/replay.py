"""Exercise the live local HTTP service using synthetic data only."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--events',default='data/sample_events.json');args=p.parse_args()
key=os.environ['FRAUDSCOPE_API_KEY'];events=json.loads(Path(args.events).read_text());latencies=[];reviews=0
for event in events:
    started=time.perf_counter()
    req=urllib.request.Request(args.url+'/v1/score',data=json.dumps(event).encode(),headers={'Content-Type':'application/json','X-API-Key':key})
    with urllib.request.urlopen(req,timeout=20) as response:r=json.load(response)
    latencies.append((time.perf_counter()-started)*1000);reviews+=r['decision']=='review'
print(json.dumps({'events':len(events),'reviews':reviews,'median_http_ms':statistics.median(latencies),'p95_http_ms':sorted(latencies)[int(.95*(len(latencies)-1))],
                  'note':'Sequential local HTTP exercise, not a production load test.'},indent=2))

