# Model card: synthetic reference champion

**Model:** logistic. **Version:** `synthetic-42-c55a5ec0ad`. **Feature schema:** pit-v1. **Production approved:** no.

## Intended use

Prioritize synthetic e-commerce transactions for an analyst review demonstration. The model is not authorized to approve, decline, or block real payments. A pass decision means below this review threshold, not verified legitimate.

## Training and selection

60,000 simulated events generated with seed 42. 17 point-in-time predictors. All preprocessing is fitted on the training stage. Isotonic calibration uses a later independent stage. Model choice and threshold minimize validation estimated cost under a 5% review target.

| Stage | Rows |
|---|---:|
| train | 31,916 |
| calibration | 8,015 |
| validation | 5,110 |
| test | 11,945 |

Purged supervised rows: 3,014. Event time is relative seconds. Labels are simulated as available two days later. Raw entity IDs and outcomes are excluded from predictors.

## Final holdout evidence

| Measure | Value |
|---|---:|
| n | 11945 |
| fraud_count | 733 |
| fraud_rate | 0.061365 |
| average_precision | 0.455098 |
| roc_auc | 0.826901 |
| brier | 0.040728 |
| log_loss | 0.164966 |
| recall_at_5pct | 0.474761 |
| value_capture_at_5pct | 0.431230 |
| review_rate | 0.041356 |
| policy_precision | 0.641700 |
| policy_recall | 0.432469 |
| false_positive_rate | 0.015787 |

Average-precision 95% day-cluster bootstrap interval: 0.4209 to 0.4930 (300 resamples). This resamples holdout days with the model fixed. Dependence across days and accounts, model-selection uncertainty and real-world transfer uncertainty are not fully captured.

## Threshold and economic assumptions

Frozen threshold: 0.511111. Review cost $3, extra legitimate-review friction $5, recovery fraction 80%. The holdout estimated avoided cost is $21,596.36 relative to no review. This is simulated, ignores platform and intervention complexity, and is not measured financial benefit. See cost_sensitivity.csv for fixed-policy alternatives.

## Failure modes and limits

Social-engineering behavior in the shifted test period can resemble legitimate transactions. Score calibration and review demand can shift. New or empty feature history can change decisions. Anonymized real-data fields require a separate availability audit. No protected-attribute fairness evaluation is possible from this simulator. The cold-start/unseen-entity slice can be small or absent and is not a guarantee for new populations.

This card describes the original tabular service champion. Version 1.1 separately fits GNN research candidates, documented in GNN_MODEL_CARD.md. No LLM is fitted. Evidence rules are observations, not model attributions. Reference sensitivity is not a causal or additive explanation.

## Artifact integrity

Data fingerprint: `c55a5ec0ad17a0bcee82cdf5d6c400e611b853d60bcab416700c7829ac67df26`

Source fingerprint: `15a50232743ea1763bfa5a3e39628abedf147767f1d9326ee83108de628c3e55`

Model file SHA-256: `33b2cbe45593f4bb90aeff95647550430d0f50230249169f74c5da5f70e5d058`

Executed with Python 3.12.14 and scikit-learn 1.8.0.

The joblib file must come from a trusted source. A checksum is not a signature. The service rejects a scikit-learn version mismatch. Keep the manifest and model together.

## Updates

No online learning or automatic promotion. Retraining, validation, release review and versioned promotion must be explicit. In real use, monitor mature-label coverage, operational workload, calibration, case outcomes and feature freshness before approving a new version.
