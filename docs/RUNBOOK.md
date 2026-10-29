# Local operation and recovery

## Start and inspect

Follow README setup, generate a local API key, and start one worker on `127.0.0.1`. The process refuses to start without the key or with an incompatible model. `/healthz` reports liveness. Authenticated `/readyz` checks loaded-model state and database access. `/v1/operations` reports stored-event count, label count, recent review fraction and stored processing-latency p95.

Stored `processing_ms` ends before the final database insert and commit. It is not end-to-end latency. `scripts/replay.py` measures local HTTP duration including the response. Neither is a throughput or availability guarantee.

The shipped service validation uses the actual application through in-process ASGI, because network and IPC listening sockets were unavailable in the build environment. Run `python scripts/validate_local_service.py` to repeat it. Live TCP serving and `scripts/replay.py` must be checked on your own machine before claiming a network-service test.

The API begins with empty history on a new database. The sample replay is a correctness demonstration, not the final holdout evaluation. A real deployment needs a controlled warm-up of prior feature history and an explicit cold-start policy. A dedicated no-decision warm-up endpoint is not implemented here.

## Input drift

```bash
python -m fraudscope.cli monitor \
  --reference reports/demo/seed_42/reference_profile.json \
  --current reports/demo/seed_42/holdout_features.csv.gz \
  --out reports/current_drift.json
```

A PSI alert suggests investigation. Inspect source schema, missingness, feature timing, segment mix, label coverage and score distribution. Do not assume drift means more fraud, and do not automatically retrain.

## Delayed feedback

Authenticated `POST /v1/feedback` accepts a confirmed label with its availability time and source. Identical retries are safe. Changed feedback returns 409 for governed correction. The service does not learn online.

```bash
python -m fraudscope.cli evaluate-feedback \
  --db runtime/fraudscope.sqlite --as-of 10368000 --lag-days 2
```

Use the same time origin as the events. The command excludes outcomes arriving after `as-of` and events that have not reached the maturity horizon. Missing outcomes stay missing. It reports per-model metrics only when both classes are observed. Current feedback may be review-selected, so those metrics are not automatically population estimates.

## Retry rules

| Response | Action |
|---|---|
| 200 | Persist the returned model version and decision in the calling application |
| 401 | Correct identity/secret configuration. Do not repeatedly retry |
| 409 changed ID | Investigate upstream deduplication or correction handling |
| 409 late event | Route to reconciliation. Do not change its timestamp to force acceptance |
| 422 | Fix the source contract or reject the event |
| 503 | Retry with bounded backoff and the same transaction ID, using the caller's approved fallback |

The reconciliation system and caller fallback are outside this reference implementation. Define them before real integration.

## Backup and integrity

Use SQLite's backup API to take a consistent backup, particularly when WAL mode is active. Do not copy only the main database file while writers are active. Restrict backups like the source data. Run `python -m fraudscope.cli audit --db runtime/fraudscope.sqlite` after restoration. The audit does not prove that the entire file was not replaced or truncated. Anchor its head hash in a separate trusted system for that purpose.

## Model promotion and rollback

Train into a new directory. Review metrics, source hash, feature schema, lineage and maturity assumptions. Preserve the old trusted model and manifest together. Set `FRAUDSCOPE_MODEL` to the approved pair and restart the service. Keep the event database. Old IDs will retain their original responses. Verify new decisions, feature-version compatibility and operational behavior before broader traffic.

The repository uses explicit file selection rather than a model registry. There is no automatic hot reload. Promotion and rollback automation, signatures and approvals are future deployment work.

## Queue capacity

The threshold targets a validation review rate, not a hard online capacity. Use `/v1/operations` to detect rising demand. A production queue needs staffing-aware admission, urgency rules and overflow handling. Silent threshold changes would invalidate the declared policy and must be evaluated and versioned.

