# Moving from inspiration to a real project

## Phase 1: establish the decision and data rights

Define the unit of prediction and the moment it is scored. An authorization-time fraud model differs from a post-settlement chargeback predictor. Document who may access the data, what may be retained, and what may appear in a portfolio. Keep authorized real data outside public repositories.

Create an outcome dictionary with fraud type, confirmed status, amount lost, amount recovered, dispute/reversal behavior, event time, outcome arrival time and intervention history. Unknown or unresolved cases are not automatically legitimate. A suspicious alert is not a fraud label.

## Phase 2: audit the feature timeline

For every column, write its source, creation time, update time and availability at scoring. Remove post-outcome indicators, revised totals, future transactions, chargeback fields and aggregates calculated over a full extract. For categorical target statistics, fit on training history with appropriate temporal folds. For graph features, build snapshots using earlier edges only.

Keep train/serve feature definitions in one module and extend parity tests with real sanitized fixtures. Do not rename IEEE columns into superficially similar synthetic fields. Create a new input contract and feature version for each operational source.

## Phase 3: replace simulation assumptions

Replace the two-day fixed delay with actual outcome-availability timestamps. Define mature cohorts and report label coverage. Add multiple time windows and a final untouched period. Report seen-entity and new-entity performance separately where identifiers support it. Account for daylight saving, time zones, multiple currencies and duplicate/corrected events.

Estimate review cost, customer friction, recoverability and false-decline harm with stakeholders. Keep a sensitivity analysis because these quantities remain uncertain. Synthetic avoided-cost figures must not appear as actual savings.

## Phase 4: run fair experiments

Begin with a simple calibrated baseline. Compare an improved tree model using the same split and tuning budget. Consider LightGBM or CatBoost only after the baseline and source contract are stable. Add transaction-only versus behavioral/network-feature ablations. Isolation Forest remains an anomaly baseline rather than a substitute for supervised validation.

Predeclare selection metrics, workload and cost assumptions. Use rolling temporal backtests and intervals appropriate to temporal and entity dependence. Retire a test set once it becomes a development target. Keep a full experiment ledger, including failed hypotheses.

For your GNN interests, a later study could model an account-device-merchant graph. Compare its added value against the simple device-degree feature. Guard against future edges, test-node labels and entity shortcuts. Measure incremental gain, inference cost, stability under perturbation and operational complexity.

## Phase 5: shadow the real workflow

Mirror eligible events through the scoring pipeline without changing decisions. Compare scores with mature outcomes and current controls. Check throughput, cold starts, missing features, unseen categories, schema changes, review capacity, and staff feedback. Record interventions so reviewed cases are not mistaken for an unbiased outcome sample.

## Phase 6: approve narrowly, then expand

The business owner, operations team, security team and appropriate reviewers must define the release criteria. Roll out narrowly with a documented fallback and reversal process. Keep model promotion separate from training. An alert or a fresh training run must not automatically replace the active model.

## Portfolio evidence you can publish

Prefer architecture, code, synthetic examples, aggregate metrics that sharing terms permit, and a clear account of limitations. Publish only permitted data and derived artifacts. Keep application credentials, real IDs, row-level predictions, transaction histories and analyst notes private.

