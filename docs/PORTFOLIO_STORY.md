# Presenting the project honestly

## Project explanation

FraudScope is a fraud review system built around time-correct data, calibrated predictions and a limited analyst budget. The reference evaluates three original model families and four added neural candidates. It serves the original selected supervised model through a validated API and monitors changes in data and review demand. The offline GNN research track adds learned relation messages and structural sensitivity experiments. A separate adapter supports local IEEE-CIS benchmarking after authorized data is supplied.

This initial reference was prepared with AI assistance as a learning foundation. Before listing it as independent experience, reproduce it, understand each component, and make documented improvements of your own.

## Five-minute demonstration

| Time | Show | Explain |
|---|---|---|
| 0:00–0:45 | Dashboard summary | Business question, synthetic scope, why accuracy is insufficient |
| 0:45–1:45 | Review-budget slider and frozen policy | Difference between ranking quality and an operational threshold |
| 1:45–2:30 | One analyst case | Observed evidence, uncertainty, human verification |
| 2:30–3:30 | Notebook chronology and feature test | Why future history and delayed labels cause leakage |
| 3:30–4:15 | API replay and duplicate retry | Contract, state, idempotency, persisted model version |
| 4:15–5:00 | Drift/capacity breach and migration plan | What failed, what is measured, what remains before deployment |

## Questions you should be able to answer

**Why did logistic regression win one run?** Selection minimizes validation estimated cost under a review constraint. A more complex model need not win that decision problem. The final test was not used to choose it.

**Is 0.477 average precision good?** It is a measured outcome for this simulator, whose holdout fraud prevalence is around 6%. It cannot be carried over to another population. Its value also depends on the usable operating point and costs.

**Why can review demand exceed 5%?** The threshold was calibrated to a validation population. Drift changes the score distribution, so a later population can exceed the target. One shipped run demonstrates this.

**Does the project use a GNN or LLM?** Version 1.1 trains three offline graph candidates and a no-edge MLP control. The deployed-reference API remains tabular. The analyst brief is deterministic with no external calls. Full HGT, recurrent TGN memory and signed/magnetic variants are future research directions. No LLM is fitted.

**Is this production ready?** It is an end-to-end, locally tested engineering reference. Real data, target infrastructure, capacity controls, privacy/security review, and operational acceptance are still required.

**What is the IEEE-CIS score?** It is pending. The adapter is tested on artificial schema fixtures. No real score has been fabricated.

## After you independently reproduce and extend it

Use claims that match your actual work. A suitable project bullet could say:

“Developed and evaluated a fraud review pipeline with chronological validation, calibrated models, point-in-time behavioral features, an idempotent scoring API, and drift monitoring. Compared models under an explicit review-capacity and cost policy.”

Add metrics only with the dataset, split and scope. Describe synthetic results as synthetic. Replace this example with your own contribution and measured evidence. Do not claim actual savings, paid deployment, customers served, a leaderboard ranking, or independently authored work that you have not completed.

