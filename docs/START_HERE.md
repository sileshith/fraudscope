# Your learning path

This project connects your mathematical background, ASU data science work, and business analytics interests. Start with the decision being made: **Which transactions deserve scarce analyst attention?** A probability score is useful only when the review policy and its costs make sense.

## 1. See the completed example

Open `reports/demo/seed_42/portfolio.html`. Move the review slider from 1% to 10%. Watch fraud capture increase, then compare it with the frozen-policy section. The slider ranks known holdout transactions retrospectively. The frozen policy uses the threshold selected before the test period. They answer different questions.

Open the case explorer. Read the evidence, including cases that eventually proved legitimate. Shared devices and large distances are signals, not verdicts.

## 2. Understand one event

Read `contracts.py`, then one object in `data/sample_events.json`. `event_time` is seconds from the simulation's origin, not a real customer's timestamp. `amount` is one assumed currency. `account_id` and `device_id` are synthetic tokens. Labels are outcomes learned later and are rejected by the scoring endpoint.

Take this example: an account has prior purchases of $20 and $30. A new $100 transaction arrives. The prior mean is $25, so `amount_ratio_7d = 100 / 25 = 4`. The current $100 is not part of the mean. If there is no prior history, the ratio is 1 and a separate count reports that the account has no observations.

## 3. Trace time before learning algorithms

Read `features.extract` and `splits.temporal_split`. On a 120-day horizon, nominal boundaries are around day 66, 84 and 96. Actual boundaries use the observed minimum and maximum event times. With a two-day maturity lag, the final two days of labels in each earlier stage are excluded from supervised use.

| Stage | Purpose | What can be fitted here? |
|---|---|---|
| Training | Learn patterns from older events | Imputers, scalers, encoders, estimators |
| Calibration | Map model scores to probabilities | Isotonic calibration only |
| Validation | Choose model and operating point | Model selection and threshold |
| Final test | Estimate later performance | Nothing |

An earlier unlabeled event may still contribute to a later velocity count. Its existence was already known, even though its outcome was not. No earlier fraud labels enter the behavior features.

Exercise: add a future transaction with an extreme amount. Earlier feature rows must not change. The test suite checks this property.

## 4. Follow the mathematics

For logistic regression, the fitted model estimates

\[p(y=1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^T x)}}.\]

The gradient boosting model can learn nonlinear interactions. Isolation Forest produces an anomaly score. A rare event can be legitimate, so anomaly detection is a comparison model rather than a direct definition of fraud.

Calibration learns a separate monotone mapping from raw scores to observed outcomes on the calibration period. A score of 0.60 means roughly 60% fraud only if calibration remains valid for the current population. The drifted test period can violate that assumption. Read the reliability plot, Brier score and log loss together.

Precision is `fraud among reviewed / all reviewed`. Recall is `fraud among reviewed / all fraud`. Accuracy can be high even if the model misses every rare fraud event, so it is not the headline measure.

Average precision is a weighted sum of precision across recall increments. The no-skill reference is the fraud prevalence. It depends on the population, so comparing AP across different datasets without prevalence and split context can mislead.

## 5. Connect predictions to cost

For transaction i, let `y_i` indicate fraud, `a_i` be its amount, and `r_i` indicate review. The illustrative loss is

\[C=\sum_i[y_i a_i(1-\rho r_i)+c_r r_i+c_f(1-y_i)r_i].\]

Here recovery `rho = 0.8`, review cost `c_r = $3`, and additional friction for a legitimate review `c_f = $5`. Reviewing a $100 fraud loses $20 plus $3 review cost. Reviewing a legitimate transaction costs $8. These are planning assumptions, not observed financial savings.

Validation selects the supervised model and threshold with the lowest estimated cost while reviewing at most 5% of validation events. The threshold stays fixed for testing. `cost_sensitivity.csv` varies review cost and recovery while holding that threshold fixed. A more advanced version could rank expected avoided loss instead of probability, then validate the policy on a new temporal period.

## 6. Run the notebook and API

Install with the README commands. Execute `python scripts/execute_notebook.py` to reproduce the notebook, including training. Use a new output directory for your own experiments so the original evidence remains intact. Start the API, replay the sample events, then resend one identical transaction. It must return the same result without adding history twice. Change its amount while keeping the ID and expect a 409 conflict.

## 7. Make a first independent improvement

Choose one question, state the expected benefit before seeing the test data, and use a fresh final holdout:

- Does an amount-aware review policy reduce cost at the same workload?
- Do device-account features help beyond transaction-only features?
- How does performance change with a 1-, 2-, 7-, or 14-day label delay?
- What happens if a device identifier is unavailable in 30% of transactions?
- Can calibration remain reliable on a later time window?

Do not repeatedly tune against the bundled final holdout and call it an untouched test set. Once inspected during development, treat it as development evidence and reserve a later period.

## 8. Move to real data

Run IEEE-CIS as a separate benchmark. Then follow `REAL_DATA_MIGRATION.md` for an operational dataset with true event timestamps, label-availability timestamps, consented fields, and documented interventions. The competition data cannot establish live feature availability or deployment suitability by itself.

## Suggested four-week study schedule

| Week | Work product | Evidence to retain |
|---|---|---|
| 1 | Reproduce the synthetic project and explain one transaction end to end | Environment, tests, notebook, personal notes |
| 2 | Run IEEE-CIS and an honest temporal baseline | Data checksums, schema, split manifest, metrics |
| 3 | Implement one predeclared improvement and evaluate on new time windows | Ablation, uncertainty, cost comparison |
| 4 | Improve operational controls and prepare a five-minute demonstration | Failure tests, runbook, architecture and project story |

Treat these as milestones, not promises about how quickly the work will be finished.

## Added GNN research track

Version 1.1 implements the temporal GNN comparison discussed in the original roadmap. Open `notebooks/02_temporal_gnn_research.ipynb`, then read `GNN_RESEARCH_DESIGN.md` and `GNN_STABILITY_PROTOCOL.md`. The no-edge neural control and relational mean-aggregation model are essential comparisons for the attention and risk-gated candidates. Their results do not establish that a newer architecture is the best operating model.

