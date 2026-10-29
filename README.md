# FraudScope: Fraud Detection with Temporal Graph Neural Networks

A fraud investigation workbench and reproducible machine-learning research reference by Sileshi Hirpa. FraudScope ranks transactions for limited analyst review, compares temporal graph models with tabular baselines, and makes evaluation and serving controls inspectable.

[GitHub](https://github.com/sileshith/fraudscope) · [Checks](https://github.com/sileshith/fraudscope/actions) · [Portfolio](https://hirpadata.com) · [Private application](https://fraudscope-sileshi.bronze-gull-3170.chatgpt.site/) · [Mastery guide](docs/MASTERY_GUIDE.md)

**All measured performance is synthetic. Real IEEE-CIS performance remains unmeasured. This is a production-oriented reference, not an approved system for real financial decisions.** The hosted app remains owner-private. ChatGPT sign-in alone does not give visitors access. Arrange an owner screen share.

## Business problem and architecture

Fraud teams have finite review capacity. False positives consume analyst time, while missed fraud carries modeled loss. FraudScope evaluates ranking within a 5% review budget separately from a frozen, validation-selected threshold and estimated cost.

The React/TypeScript [workbench](applications/workbench) runs on Cloudflare Workers with D1 storage. It supports a transaction queue, evidence, historical account/device links, saved reviews, CSV exports, dashboards and audit verification. An active calibrated logistic model supplies saved review decisions. Risk-gated temporal attention supplies **shadow scores**. Four neural models can be compared for a case without changing its saved decision. No payment processor is connected and no payment is automatically blocked.

The Python package provides simulation, point-in-time features, chronological evaluation, calibration, graph experiments, structural stress analysis, a FastAPI reference and an IEEE-CIS adapter. Offline research candidates do not automatically become the live decision model.

## Data, evaluation and synthetic results

The recorded comparison uses three synthetic populations of 60,000 transactions each over 120 days, with seeds 42, 73 and 101. Features use information available before each transaction. Training, calibration, validation and testing use 55/15/10/20% of elapsed time, with purges for an assumed two-day label delay. The delay, 5% budget and costs are experiment assumptions.

Baselines are logistic regression, histogram gradient boosting and Isolation Forest. Neural models are a no-edge MLP, relational GraphSAGE-style aggregation, temporal relation attention and risk-gated temporal attention. The graph is a one-hop transaction projection with account/device relations, historical windows and bounded neighborhoods. Full HGT and persistent-memory TGN are research directions, not implemented architectures.

| Model | Mean AP | Mean recall within top 5% |
|---|---:|---:|
| Gradient boosting | 0.4855 | 50.69% |
| No-edge MLP | 0.4942 | 52.84% |
| Relational GraphSAGE-style aggregation | 0.4974 | 53.95% |

Source: [deployed evidence export](applications/workbench/model-data/evidence.json) and [recovered summary](reports/gnn/multi_seed_summary.json). These descriptive three-seed means do not establish statistical significance or real-data performance. **No GNN won overall validation cost:** the MLP won seeds 42/73 and gradient boosting won seed 101. The workbench retains its original logistic policy.

Structural tests distinguish graph changes, embedding drift, score drift and threshold flips with frozen weights, calibration and thresholds. The deployment contains 12 raw seed-42 stress draws. Historical three-seed chart means are preserved separately with [recovery provenance](reports/recovery/structural_chart.json). Original raw three-seed stress files and Python checkpoints were not recovered. Research commands below regenerate them.

## Verify the workbench without retraining

Use **Node 22.13 or later** and **pnpm 11.25.0**, as declared by the project. Install the pinned package manager through your existing Node installation if needed. The local adapter uses Node's SQLite API.

```bash
cd applications/workbench
pnpm install --frozen-lockfile
cp -n .openai/hosting.example.json .openai/hosting.json
pnpm test
pnpm run check
pnpm run review:verify
```

Expected: 12 application tests pass, TypeScript exits successfully, and `REVIEWER TOUR PASSED` reports 24 transactions and 25 audit records. Tests include 80 model-parity cases and a 48-event independent Python parity replay. The tour exercises the API with isolated SQLite storage. It does not test hosted sign-in, target-scale Cloudflare D1 or network performance.

The [private app](https://fraudscope-sileshi.bronze-gull-3170.chatgpt.site/) and [handbook](https://fraudscope-sileshi.bronze-gull-3170.chatgpt.site/handbook) require owner-authorized access. New deployments need their own Site identity and storage setup. Public source contains a hosting example without a personal Site ID.

## Research reproduction

From the repository root, use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.dev.lock
python -m pip install -e . --no-deps
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python -m pytest
python -m fraudscope.cli demo --rows 60000 --seeds 42 73 101
```

Fresh recovery checks on the Intel Mac passed **41 tests, with one module skipped** because PyTorch was unavailable. For the locked GNN environment, use Linux CPU, install `requirements.gnn.lock`, then run:

```bash
python -m pytest
python scripts/run_gnn.py --rows 60000 --seeds 42 73 101 --epochs 12
```

The GNN lock does not install on an Intel Mac as written. Read [GNN_QUICKSTART.md](docs/GNN_QUICKSTART.md). The full historical 50-check suite and new GNN training were not rerun during recovery.

The tabular command generates the model needed by the FastAPI reference. Load only trusted joblib files in compatible environments. The workbench includes exported model weights and requires no Python training for local verification. Model export scripts expecting Python checkpoints require regenerated reports first.

For IEEE-CIS, place authorized transaction and optional identity CSVs under `data/ieee`, then run `python -m fraudscope.cli benchmark --data data/ieee --out reports/ieee --label-lag-days 2`. The adapter joins on `TransactionID`. An artificial schema test does not establish real IEEE-CIS performance. Read the [benchmark protocol](docs/IEEE_BENCHMARK.md).

## Review and release boundary

Start with the [hiring-manager walkthrough](docs/HIRING_MANAGER_REVIEW.md), [mastery guide](docs/MASTERY_GUIDE.md), [PowerPoint](presentations/FraudScope_Employer_Presentation.pptx) and [matching PDF](presentations/FraudScope_Employer_Presentation.pdf).

This release recovers deployed source commit `cfe59beffa54d7693026c5ac7b1668e5adb9971a` and its included Python research. It adds a freshly verified local tour, documentation and presentation exports. The original Library ZIP could not be downloaded through the connected browser. This is **not a byte-for-byte recovery of that complete archive**. Read [recovery evidence and gaps](docs/RECOVERY_STATUS.md). Historical validation files retain their original meaning and do not substitute for fresh checks.

Real deployment still requires authorized data, feature-timing validation, load tests, recovery drills, external audit anchoring and independent security/model-risk review. Internal hashes cannot detect a complete privileged rewrite without an external anchor. See [production readiness](docs/PRODUCTION_READINESS.md).

This is an AI-assisted reference. Present work you can independently explain and reproduce. Original code and synthetic examples use the included [MIT license](LICENSE). Dependencies and datasets retain their terms. See [primary sources](docs/SOURCES.md).
