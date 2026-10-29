# IEEE-CIS benchmark protocol

## Current status

**Real benchmark: not run.** The source data was not supplied. No IEEE-CIS metric or leaderboard rank is claimed. The adapter is tested with artificial files that match selected column names. Those tests validate joins, schemas, exclusions, and pipeline execution only.

Official sources: [dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data), [competition](https://www.kaggle.com/competitions/ieee-fraud-detection), [evaluation](https://www.kaggle.com/competitions/ieee-fraud-detection/overview/evaluation). Review the current terms yourself before downloading or redistributing data.

## Inputs and preparation

Use `train_transaction.csv` and preferably `train_identity.csv`. The label is `isFraud`. `TransactionDT` is a relative time field. Join identity records with a left, one-to-one merge on `TransactionID`, preserving transactions without a match. Fail on duplicate or null keys, invalid amounts, nonbinary targets, or overlapping identity fields.

The adapter uses an allowlist for the original transaction and identity feature families. It excludes the target, transaction ID, raw time, and arbitrary additional columns. It treats card/address codes as categories, along with string fields and identity codes from `id_12` onward. Numeric features are imputed and scaled, categorical features are imputed and one-hot encoded with training-derived grouping and a maximum of 32 categories per field. This is a memory-conscious baseline rather than a competition-winning feature recipe.

The default full dataset may require several gigabytes of RAM after dense encoding and may take longer than the synthetic experiment. Start on a machine with adequate memory and monitor usage. No full-data runtime or memory measurement is claimed here. Do not silently subsample and compare the result with full-data reports.

## Protocol

```bash
python -m fraudscope.cli benchmark --data data/ieee --out reports/ieee --seed 42 --label-lag-days 2
```

Sort by `TransactionDT`, breaking ties with `TransactionID`. Allocate elapsed time to training/calibration/validation/test at 55/15/10/20%. Fit preprocessing and models on training only, calibrate on the next stage, choose model and threshold on validation, then freeze both for test. Do not use the unlabeled competition test set to fit encoders, scalers, calibrators or target statistics.

Two days is an assumed outcome delay for a reproducible scenario, not a verified dataset fact. Repeat with 0, 7 and 14 days as a predeclared sensitivity study. If repeating after inspecting the test, label the study exploratory or reserve another final period.

Report ROC-AUC for compatibility with the competition's metric and average precision, prevalence, calibration measures, top-5% recall and value capture for business interpretation. Local temporal results are not leaderboard results. The supplied workflow does not make a Kaggle submission or estimate hidden-test performance.

## Comparison rules

1. Train new models for the IEEE schema. Never apply the synthetic event model directly to competition rows.
2. Compare candidate models on exactly the same rows, split boundaries, label assumptions and target definition.
3. Keep the competition and synthetic tables separate, even when they use the same metric names.
4. Include row counts, positive counts, time boundaries, source checksum, versions, seed and feature list.
5. Repeated customer proxies may occur across time splits. This tests future transactions in an ongoing system, not entirely new customers. A separate grouped or unseen-entity experiment needs a justified identity proxy and explicit leakage analysis.
6. Vesta-derived anonymized aggregates may contain information whose timing cannot be audited. Document that uncertainty. A time split reduces evaluation leakage but does not prove each field was available at authorization.
7. Calibrate and select thresholds without touching the final test. Published scores from different splits are not apples-to-apples benchmarks.
8. Derived real-data artifacts, labels and examples may be restricted too. Keep the default ignored `reports/ieee/` output private unless sharing is permitted.

## Results table to complete after your run

| Evidence | Value |
|---|---|
| Authorized dataset version and checksum | Pending |
| Local split sizes and fraud prevalence | Pending |
| ROC-AUC / AP / Brier / log loss | Pending |
| Fixed-threshold review rate and fraud recall | Pending |
| Recall and value capture at 5% | Pending |
| Cost assumptions and sensitivity | Pending |
| Uncertainty and seed stability | Pending |
| Feature availability audit | Pending |

