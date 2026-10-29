# FraudScope deployed application

This release serves a private full-stack application. The server runs the trained logistic decision model and risk-gated GNN in shadow. It persists transactions, historical feature vectors, reviews and audit records in Cloudflare D1. All data used to build and validate the models is synthetic.

## Access and ownership

The hosting platform controls ChatGPT sign-in and owner-private access. Each user-data API also requires the platform-injected user identifier. Every record lookup and write is scoped to that identifier. The health and research routes expose only operational status and synthetic research evidence within the private Site boundary. Never place this Worker behind a proxy that permits callers to forge platform identity headers.

There is no payment processing, automatic payment block, credit decision or real customer-data ingestion. Analyst judgments are distinct from verified fraud labels. They are not used for automatic training or model promotion.

## Deployed model behavior

The active decision model is the original seed-42 logistic pipeline with its held-out isotonic calibrator and validation-selected threshold. The GNN shadow is the original risk-gated temporal attention model, exported as weights and evaluated in JavaScript. Four neural models can be compared for an individual case. They use the stored historical feature encoding and past-only teacher scores.

No weights are invented, distilled or refitted during export. `scripts/export_models.py` rejects unsupported preprocessing configurations. `model-data/parity-report.json` records comparison against independent Python outputs. The standalone JavaScript arithmetic uses double precision and an error-function approximation for GELU, so numerical tolerance is explicit.

The workspace starts after the original training period. A frozen training-only logistic teacher scores new events. Entity history starts empty and grows only from submitted workspace events. Cold starts and this constructed replay stream differ from the research holdout population. The application does not claim that live demonstration scores reproduce holdout performance.

## Persistence and concurrency

A workspace revision serializes scoring and review writes. Each atomic D1 batch first compares and advances that revision. Dependent inserts require a unique operation token from the successful update. Concurrent losers recompute from fresh history or receive a conflict. An identical transaction ID and payload returns the saved response. Reusing the ID with different content fails. Strictly earlier timestamps enter features and graph messages. Equal-time events cannot see one another.

Reviews carry their own expected revision to prevent lost edits. Audit records form a SHA-256 chain that also checks stored scoring artifacts and review state. This provides internal tamper evidence. It cannot prove integrity after a complete privileged rewrite without an external trusted anchor.

The reference workspace is capped at 12,000 committed actions. Replay adds up to 240 constructed events, four per HTTP call. These bounds keep query and resource usage reviewable. The UI reports server scoring computation separately from end-to-end request latency. Structured request logs omit raw transaction fields and notes.

## API

- `POST /api/workspace` creates the signed-in user's workspace if absent.
- `POST /api/score` validates and scores one raw transaction.
- `POST /api/replay` adds one to four constructed synthetic events.
- `GET /api/overview` summarizes the persisted workspace.
- `GET /api/transactions` supports filters, search and cursor pagination.
- `GET /api/transactions/:id` provides the case evidence and historical graph.
- `POST /api/transactions/:id/review` saves a revision-checked analyst judgment.
- `GET /api/transactions/:id/models` computes all four neural candidates.
- `GET /api/research` provides the recorded synthetic experiment results.
- `GET /api/audit` and `POST /api/audit/verify` inspect audit integrity.
- `GET /api/export` exports CSV with formula-leading cells escaped.
- `GET /api/health` checks database schema availability and both active and graph inference.

Write requests reject foreign origins, cross-site fetches, unknown fields, nonfinite numbers, out-of-contract values, invalid JSON and oversized bodies. Runtime failures return a request identifier without raw internal details. The private platform access policy is the primary boundary. Public exposure needs a separate access, rate-limit and abuse review.

## Reproduction and release

Use Node 22 or newer and the included lockfile. Install dependencies with the Sites install helper in this environment or the package manager outside it. Run `node --test tests/*.test.mjs`, `node node_modules/typescript/bin/tsc --noEmit`, and the supported Sites build workflow. The schema lives in `db/schema.ts`. SQL migrations under `drizzle/` are applied by the hosting release process. Preserve already-applied migrations and append future changes.

The `research/` directory contains the original Python model and dataset-generation source with pinned dependency inputs. Run that workflow to regenerate training artifacts before running `scripts/export_models.py --research-root research`. Exact original evidence and trained checkpoints remain in the separately delivered FraudScope research archive.

## Before real financial production

Obtain and validate authorized data, observed label-availability timestamps, identity semantics and business costs. Reserve a fresh temporal evaluation period. Run load and concurrency tests on the target hosting plan, set an agreed service-level objective and alert routing, establish database backup and recovery, anchor audit heads externally, verify retention and deletion controls, and rehearse rollback. Add role-based review authorization if access expands beyond the owner. Independently review security, privacy and model risk.

The existence of a deployed URL does not complete these real-data and operational gates. No financial production certification is claimed.
