# Structural stability study

## Implemented: frozen-model diagnostic

For each simulation seed, choose the graph candidate by validation estimated cost. Freeze its weights, calibrator and review threshold. Perturb only incoming edges of final-test transactions. Earlier graph rows and all transaction features stay fixed. No model is retrained during this study.

Budgets are 10% and 30% of eligible test incoming edges. Each budget has three independent perturbation draws for deletion and rewiring. The log reports requested and actual changes. Rewiring chooses a strictly earlier source with no duplicate source within that target relation when possible. It retains the relation label and target indegree, but it does not preserve the original entity membership, original time-window bound or source outdegree. Rewired ages are recomputed. This is an artificial integrity stress, not a realistic fraud simulator.

## Recorded quantities

| Quantity | Definition | Interpretation |
|---|---|---|
| Raw adjacency displacement | Frobenius norm across binary relation adjacency matrices | Square root of the number of changed entries in the final binary adjacency. Rewiring operations can restore earlier removed edges, so operation count alone is not sufficient. |
| Embedding drift | Mean Euclidean norm between the model's final pre-head representations | Sensitivity within the same model and representation scale |
| Score drift | Mean absolute calibrated score difference | Change in the selected model's output |
| Decision flip rate | Fraction crossing the original threshold after perturbation | Policy instability at this operating point |
| AP, ROC-AUC, review/cost metrics | Re-evaluated with the original test labels | Diagnostic performance change under corrupted structure |

The adjacency norm is not the displacement of the learned attention operator. Embedding drift is not normalized across architectures. Calibration can create ties and amplify threshold flips. Each interpretation needs these limits.

## Implemented guardrails

- Graph validation rejects future and simultaneous messages.
- Labels and audit-only scenario fields do not form graph features or edge signs.
- Teacher training uses only earlier mature labels.
- No-edge MLP predictions are invariant to adjacency changes.
- A graph-input change can alter the GNN score, verifying that messages reach the classifier.
- Checkpoint reload reproduces scores.

## Next studies, not yet implemented

1. Paired temporal backtests and at least five independent data seeds, with uncertainty on the GNN-minus-MLP difference.
2. Transaction-only feature ablation to isolate engineered behavior from learned message passing.
3. Degree-aware edge addition and entity-identity corruption matched to a plausible operational failure.
4. Matched retraining at each corruption budget, clearly separated from frozen response.
5. Full HGT or TGN memory models under the same event-time contract and comparable compute budgets.
6. Signed/magnetic operators only after defining valid signs and directions, followed by charge and normalization studies.

This protocol creates a bridge to signed/directed GNN stability research. The current experiments do not prove robustness, certify an intervention policy or establish a theorem for the new fraud setting.
