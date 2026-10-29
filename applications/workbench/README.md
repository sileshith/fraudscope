# FraudScope workbench

The React/Workers application for **FraudScope: Fraud Detection with Temporal Graph Neural Networks**. Active logistic decisions and risk-gated GNN shadow scoring remain distinct. All performance is synthetic.

From this directory, use Node 22.13+ and pnpm 11.25.0:

```bash
pnpm install --frozen-lockfile
cp -n .openai/hosting.example.json .openai/hosting.json
pnpm test
pnpm run check
pnpm run review:verify
```

The local tour uses isolated SQLite storage, not the hosted identity gateway. It verifies 24 events, four neural comparisons, retries, isolation, a saved review, CSV export and 25 audit records. Application tests include independent Python parity fixtures.

The deployed application remains owner-private. New deployments require their own Site identity and D1 setup. The hosting example has no personal Site ID. See [deployment documentation](docs/DEPLOYMENT.md), the repository [README](../../README.md) and [review guide](../../docs/HIRING_MANAGER_REVIEW.md).

The Python research package is at the repository root, two directories above this workbench. Regenerate Python models and GNN checkpoints there before running checkpoint-dependent model export scripts. To export a new compatible set after training, use `PYTHONPATH=../.. python scripts/export_models.py --research-root ../..`. Preserve current evidence and review model promotion separately; do not replace the active policy merely because a candidate ranks well.

The public source-download ZIP contains a sanitized snapshot without live Site identity, dependency folders or runtime databases. Historical validation JSON files preserve their original reporting scope. Fresh recovery checks and missing original-archive artifacts are documented in [RECOVERY_STATUS.md](../../docs/RECOVERY_STATUS.md).
