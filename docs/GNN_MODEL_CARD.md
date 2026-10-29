# Temporal GNN research model card

Version 1.1.0, 26 October 2026. This card describes optional offline graph candidates. The original scoring service remains on its v1 tabular model. No graph model has been approved for production.

## Intended purpose and implementation

Study whether learned messages from earlier account-linked and device-linked transactions improve fraud ranking and review decisions beyond engineered behavior. The graph is a directed, relation-typed projection onto transaction nodes, with up to eight neighbors per relation. Account history spans seven days and device history one day. Equal-time and future messages are forbidden. Missing identifiers do not create shared nodes.

Four original PyTorch models share a 48-dimensional current-feature encoder and classifier: a no-edge MLP control, relation-specific mean aggregation, four-head temporal attention, and risk-gated temporal attention. The no-edge control retains engineered behavioral and graph-count features; it is not a transaction-only ablation. The implementation is inspired by, but does not reproduce, GraphSAGE, HGT, TGN, BRIGHT or RGH-GNN. It does not implement full heterogeneous entity states, recurrent event memory, or signed/magnetic operators.

Risk guidance comes from five expanding chronological logistic-teacher blocks. Every fitted teacher uses only labels already available before its scoring block. The initial block uses a fixed 0.02 prior. Later periods use a teacher frozen on mature training labels. Teacher risk is not a ground-truth label or edge sign.

## Evidence and selection

All measured results below use synthetic data: 60,000 events per seed, seeds 42, 73 and 101. The same data fingerprints, features and chronological stages are used for tabular and neural comparisons. Neural fits use 12 fixed epochs, batch size 512, AdamW learning rate 0.001, weight decay 0.001, dropout 0.1 and gradient clipping at 1. Calibration uses the calibration period; cost and threshold selection use validation; the final period is held out for this fit.

| Family | Mean average precision | Sample SD of AP | Mean recall at top 5% |
|---|---:|---:|---:|
| Logistic | 0.4516 | 0.0206 | 47.13% |
| Gradient boosting | 0.4855 | 0.0233 | 50.69% |
| Isolation Forest | 0.3422 | 0.0148 | 34.47% |
| No-edge MLP | 0.4942 | 0.0266 | 52.84% |
| Relational mean aggregation | 0.4974 | 0.0235 | 53.95% |
| Temporal attention | 0.4929 | 0.0268 | 53.54% |
| Risk-gated attention | 0.4941 | 0.0251 | 53.47% |

Relational mean aggregation has the highest mean ranking metrics in this pilot. The overall validation-cost choices are the no-edge MLP for seeds 42 and 73 and gradient boosting for seed 101. None is a GNN. Temporal attention is selected within the graph-only candidates in all three runs and is used for structural stress. The risk-gated model is included because it is a relevant research hypothesis, not because this pilot establishes superiority.

The GNN study extends a synthetic benchmark whose v1 test results had already been inspected. It is exploratory, not a fresh blinded confirmation. Across-seed sample SD is not a confidence interval for improvement. Per-model day-bootstrap AP intervals are included, but there is no paired significance claim. Reserve a fresh time period and predeclare model choices for confirmatory work.

## Stability evidence

For the validation-selected graph model, delete or causally rewire 10% and 30% of incoming test edges, with three draws per condition. Keep weights, features, calibrator and threshold frozen. Report exact final binary-adjacency displacement, representation drift, score drift, decision flips and predictive metrics. Rewiring can break entity membership and original relation time windows. These are controlled integrity stresses, not validated attacks or robustness certification. See GNN_STABILITY_PROTOCOL.md.

## Benchmark and operational boundaries

The optional IEEE-CIS adapter uses weak composite proxies: card1/card2/addr1 and DeviceInfo/id_31. They are anonymized attribute groups, not verified account or device identities. Real IEEE-CIS files were not supplied and no real benchmark score is claimed. An artificial-schema smoke run exercises the graph route. The assumed two-day label lag must be tested against actual outcome timestamps.

No online graph feature store, incremental neighborhood service, graph-specific latency/load test, distributed state, recurrent memory, fresh-label feedback retraining or production model promotion is provided. The service's tabular parity checks do not validate graph serving. Those are separate release gates.

Synthetic fraud prevalence, entity reuse, scenario mechanisms, delayed labels and distribution shifts are designed assumptions. Their artifacts can favor particular architectures. Protected attributes and real subgroup outcomes are absent, so no fairness claim is supported. Review costs and recovery assumptions are illustrative.

## Reproduction and provenance

Use docs/GNN_QUICKSTART.md and notebooks/02_temporal_gnn_research.ipynb. Each run records data/source fingerprints, split boundaries, a pre-fit protocol, teacher fold boundaries, validation selection, training curves, state-dict weights and checksums. Reloaded checkpoints reproduce scores. Only load trusted bundled joblib preprocessing and calibration files; a checksum is not trusted authorship.

Linux CPU Python 3.12 and PyTorch 2.14.1+cpu were used. No GPU or paid AI service is required. Exact target-platform compatibility must be checked independently. See reports/VALIDATION.md for executed checks and pending deployment work.
