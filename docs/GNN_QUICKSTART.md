# Run the GNN extension

Read GNN_RESEARCH_DESIGN.md first. The tabular setup remains usable without PyTorch. The GNN add-on was executed on Linux CPU with Python 3.12 and PyTorch 2.14.1+cpu.

## Environment

```bash
python -m pip install -r requirements.dev.lock
python -m pip install -r requirements.gnn.lock
python -m pip install -e . --no-deps
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python -m pytest
```

The exact GNN lock targets Linux CPU. On your Intel iMac, use a Linux environment rather than forcing these wheels into the existing macOS environment. PyTorch's official macOS x86 binaries stopped after the 2.2 series. A Linux container or a suitable remote Linux environment can run this CPU reference. Container and remote-job execution are not claimed as tested here. Do not replace your working ASU or research environments merely to try this package.

## Reproduce the recorded experiment

```bash
python scripts/run_gnn.py --rows 60000 --seeds 42 73 101 --epochs 12
```

This regenerates the same three synthetic populations, verifies their baseline fingerprints, constructs causal relation graphs, trains four neural candidates per seed, calibrates them, selects policies and performs structural stress tests. Existing tabular results are retained. The service model is not changed.

For a smaller first run, use a new baseline and output directory:

```bash
python scripts/run_gnn.py --rows 12000 --seeds 42 --epochs 3 \
  --baseline-root reports/gnn_smoke_baselines --out reports/gnn_smoke
```

Using a different row count against an existing baseline directory is rejected because the fingerprints differ. Do not overwrite the recorded evidence when making experiments.

## Read the evidence

- `reports/gnn/multi_seed_summary.json`: all seven model families on matching splits.
- `reports/gnn/seed_42/gnn_report.html`: readable model and stress comparison.
- `reports/gnn/seed_42/protocol.json`: fixed configuration recorded before fitting.
- `teacher_folds.json`: causal teacher-label timing audit.
- `causal_graph.npz`: relation-neighbor indices, time differences and historical teacher scores.
- `*.pt` and matching JSON files: neural state dictionaries and integrity metadata.
- `*_policy.joblib` and `preprocessing.joblib`: trusted local preprocessing/calibration objects.
- `structural_stress.json`: individual perturbation draws and outcome changes.
- `notebooks/02_temporal_gnn_research.ipynb`: executed walkthrough and checkpoint replay.

Neural weights load with `weights_only=True`. The accompanying joblib objects still require a trusted source. Checksums do not establish authenticity. No customer IDs or real benchmark data are bundled.

## Research boundaries

Only the synthetic graph pipeline has been trained and measured. The IEEE-shaped GNN route passed a 3,000-row artificial fixture with all four neural models. Real IEEE-CIS training and full-scale resource checks still await authorized files. The API has no causal graph state service or GNN promotion path. Implement and validate graph serving, history initialization, drift monitoring and rollback before making real-time GNN claims.
