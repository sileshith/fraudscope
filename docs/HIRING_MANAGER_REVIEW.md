# Hiring-manager verification

## First clone on your Mac

Use Terminal with Git, Node 22.13+ and Python 3.12 available. Keep this project separate from customer-churn-prediction.

```bash
mkdir -p ~/Desktop/ASU/projects
cd ~/Desktop/ASU/projects
git clone https://github.com/sileshith/fraudscope.git
cd fraudscope
git branch --show-current
git rev-parse HEAD
```

Expect branch `main`. Record the printed commit and compare it with the GitHub commit you are reviewing. If `fraudscope` already exists, use the update procedure below instead of cloning into it.

A public HTTPS clone does not need GitHub CLI authentication. If you use `gh`, reuse your configuration location in each new terminal:

```bash
export GH_CONFIG_DIR="$HOME/Library/Application Support/GitHub CLI"
/usr/local/bin/gh auth status
```

This recovery session found that the saved CLI login was invalid, while the connected GitHub tool had write access. If CLI authentication is needed, run `gh auth login --hostname github.com --git-protocol https --web` yourself. Never send tokens or login codes to chat. The export setting applies only to that terminal unless you choose to add it to your shell configuration.

## Update an existing checkout without losing work

```bash
cd ~/Desktop/ASU/projects/fraudscope
git remote get-url origin
git status --short
git branch --show-current
git fetch origin
git log --oneline HEAD..origin/main
```

Confirm origin is `https://github.com/sileshith/fraudscope.git` or its equivalent SSH URL. If status lists changes, stop before pulling. Commit them on a personal branch or make a separate backup. An optional stash is `git stash push -u -m "Before FraudScope update"`; inspect and restore it later with `git stash list` and `git stash apply`. Applying may require conflict resolution. Avoid reset-hard and force-pull recipes.

Once the checkout is clean and you are on `main`:

```bash
git pull --ff-only origin main
git rev-parse HEAD
```

If fast-forward fails, preserve both histories and inspect the divergence. Do not discard local commits.

## Ten-minute technical review

From the repository root:

```bash
node --version
cd applications/workbench
npm install --global pnpm@11.25.0
pnpm --version
pnpm install --frozen-lockfile
cp -n .openai/hosting.example.json .openai/hosting.json
pnpm test
pnpm run check
pnpm run review:verify
```

If your Node manager already supplies pnpm 11.25.0, skip the global installation. Use your existing Node manager if global installation reports a permissions error. Do not use sudo merely to work around that error. `cp -n` preserves any existing local hosting configuration.

Expected outputs: 12 passing application tests, no TypeScript errors, and `REVIEWER TOUR PASSED`. The tour prints 24 events, four neural comparisons, 24 CSV rows and 25 valid audit records. It checks exact retries, conflicting retries, isolation and stale-review rejection. It uses an isolated local SQLite adapter and never contacts the production workspace. Read its source at `scripts/review-verify.mjs`.

Review `model-data/provenance.json`, `parity-report.json`, the two fixture files, and `lib/inference.mjs`. Bundled neural weights run in JavaScript; no GPU or Python training is needed for these checks.

## Three-minute employer demonstration

Arrange an owner screen share of the private application. A hiring manager cannot enter merely by signing into ChatGPT. Do not change access to make the demonstration easier.

1. Explain the synthetic-data boundary and show the active logistic policy versus GNN shadow scoring.
2. Use Replay demo feed. Repeated clicks advance the bounded feed; they do not reset history. Existing workspace counts may differ from the isolated 24-event tour.
3. Open a transaction, explain its evidence and inspect historical account/device neighbors. A shared device is a relationship proxy, not proof of coordinated fraud.
4. Compare neural scores, save a review and export CSV. Analyst assessments remain unverified labels.
5. Show audit verification and the research results. Explain why higher recall at a fixed review budget did not make a GNN the lowest-cost validation choice.

If hosted access fails, run the local tour and show the PowerPoint/PDF. This demonstrates backend behavior but does not substitute for a live hosted browser test.

## Research review

Return to the repository root and follow README research setup. Run the tabular demo to regenerate the Python service model and reports. Use a separate Linux CPU environment for the locked GNN run. The full run uses 60,000 rows per seed, seeds 42/73/101 and 12 epochs. A smaller smoke run is useful for learning but does not reproduce the published metric scope.

Read `fraudscope/features.py`, `splits.py`, `graph_data.py`, `graph_teacher.py`, `gnn.py` and `graph_experiment.py`. Verify that transformations fit only on training data, historical risks use matured past labels, and thresholds freeze before testing. IEEE-CIS CSVs are user-supplied and no real benchmark result is included.

## What a reviewer should challenge

Ask why a no-edge MLP is required, why calibration is separate from threshold selection, and whether a future fixed threshold can guarantee a review budget. Ask which checks run locally, which rely on historical evidence and which require Linux or hosted access. The recovery status document records those boundaries explicitly.
