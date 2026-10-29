# FraudScope GNN research extension

## Decision and scope

Add temporal relational GNNs to the existing fraud models. Make the graph contribution measurable with a no-edge neural control, a simple mean-aggregation baseline, and structural stress tests. The shipped implementation learns from neighboring transactions, so it goes beyond using graph summary features in a tabular model.

This is an original reference implementation informed by research through 25 October 2026. It is not claimed as the newest or best fraud architecture, a reproduction of a published model, or a production GNN service. The initial API remains on the tabular model until graph serving is independently validated.

## Research choices

| Direction | Why it matters | Status in this project |
|---|---|---|
| Temporal relational message passing | Transactions can share devices/accounts, but only earlier information is available at scoring | Implemented |
| Typed, relative-time attention | Relation meaning and age can change a neighbor's relevance | Implemented with four attention heads |
| Risk-guided message gates | Historical model risk can help preserve rare suspicious signals among ordinary neighbors | Implemented with a strictly past-trained logistic teacher |
| Full HGT entity representations | Explicitly learn distinct transaction/account/device node states | Future candidate, not implemented |
| TGN persistent memory | Maintain learned entity state as events arrive | Future candidate, not implemented |
| Signed or magnetic GNNs | Study meaningful signed/directed relations and charge sensitivity | Research extension requiring defensible graph semantics |

The October 2026 RGH-GNN preprint uses supervised risk scores to guide graph aggregation. Its stratified split and bidirectional construction differ from this project's strict historical protocol. We adopt the risk-guidance idea, not its reported performance or evaluation design. We use a simpler logistic teacher instead of its LightGBM pipeline. See SOURCES.md for primary references.

## Graph representation

Conceptually, transactions connect to account and device entities. The implementation projects those two incidence relations onto transaction nodes:

\[
\mathcal N_r(i)=\{j:t_j<t_i,\;\operatorname{key}_r(j)=\operatorname{key}_r(i),\;t_i-w_r<t_j\}.
\]

Use the latest eight earlier neighbors per relation. Account history spans seven days. Device history spans one day. Edges carry information from historical transaction j to the current transaction i. Equal-time events do not observe each other, even when one row appears first. Missing keys form no edges. There are no learned entity-ID embeddings, so a new entity does not require an existing vocabulary entry.

This is a relation-typed transaction projection of a heterogeneous system. It is not a full heterogeneous-node HGT. One projected edge represents the path transaction-to-entity-to-transaction. The graph network uses one message-passing layer, not a deep recurrent memory system.

## Message passing

The shared encoder maps transaction features to a 48-dimensional representation. Each relation has its own key and value projection. Four attention heads compare the current transaction with earlier neighbors, including an encoding of `log(1 + age_hours)`. For a relation r:

\[
\alpha_{ij}^{(r)}=\operatorname{softmax}_{j\in\mathcal N_r(i)}
\left(\frac{(Qh_i)^\top[K_rh_j+\phi(\Delta t_{ij})]}{\sqrt{d_h}}\right).
\]

The risk-gated variant changes the message values:

\[
g_{jr}=0.1+0.9\,\sigma\left(\operatorname{softplus}(a_r)[s_j-\sigma(b_r)]\right),
\qquad m_i^{(r)}=\sum_j\alpha_{ij}^{(r)}g_{jr}V_rh_j.
\]

The classifier receives the current encoding, relation messages and neighborhood coverage. It is trained with binary cross-entropy, fixed 12 epochs, AdamW, gradient clipping and fixed seeds. Calibration and threshold fitting happen later, on separate periods. The fixed epoch count is a pilot budget, not evidence that the architectures were exhaustively tuned.

## Four new candidates

1. **No-edge MLP.** The same current-feature encoder and classifier structure, with zero graph messages and zero graph coverage. This tests whether nonlinear transaction features explain apparent gains. Graph branches are inactive.
2. **Relational GraphSAGE-style model.** Mean aggregation through relation-specific projections. It is a compact one-hop baseline, not a library-exact reproduction of every GraphSAGE variant.
3. **Temporal relational attention.** Typed multi-head attention with relative-time encoding.
4. **Risk-gated temporal attention.** The same attention model with earlier teacher-risk gates.

All use the same input features and temporal stages. The existing logistic regression, histogram gradient boosting and Isolation Forest remain in the comparison. The no-edge MLP still receives the original engineered behavioral features, including device degree. It removes learned neighbor messages, not every possible graph-derived signal. A strict transaction-only feature ablation is a separate future study.

## Preventing leakage from a supervised teacher

A risk-gated graph can accidentally transmit labels through in-sample teacher predictions. Here, five expanding chronological blocks generate training-period scores. Each teacher fits only labels whose availability precedes the block start, with its own past-only preprocessing. The earliest block uses a fixed 0.02 prior. The later calibration, validation and test events are scored by a teacher trained only on the mature training partition.

Historical event risk is retained as it would have been produced at that historical stage. Graph labels are never inputs. The manifest records each teacher's maximum fitted label time. Unit tests verify that changing nontraining labels does not change teacher scores.

## Evaluation and selection

The three original seeds are regenerated exactly, and their data fingerprints and temporal splits must match the saved tabular baselines. There is no oversampling across stages. Each neural model is calibrated separately and receives its own validation-selected threshold. Overall candidate choice is based on validation estimated cost, not final test AP.

The added models are a research extension after the initial synthetic benchmark was inspected. Their fixed configuration avoids further test tuning, but the results remain exploratory. A publication-quality claim needs new holdout periods, more seeds, temporal backtests, fair tuning budgets, paired uncertainty and real datasets. An apparently better average alone does not establish significance.

## IEEE-CIS graph path

The graph adapter uses composite `card1 + card2 + addr1` and `DeviceInfo + id_31` keys as provisional relations. These are noisy attribute proxies, not verified account or device identities. Missing or unknown components do not create a shared node. Shared browser/device descriptions can still generate broad, weak links. Audit key uniqueness, collision rates, degree distributions and feature availability before interpreting relational gains.

```bash
python scripts/run_gnn.py --ieee-data data/ieee \
  --out reports/ieee_gnn --seeds 42 --label-lag-days 2
```

This trains a new model using the IEEE feature schema. The real dataset has not been run. The full dense feature matrix and tensor gathers can require substantial memory. The two-day outcome delay remains an assumption. The graph adapter does not automatically create an IEEE-compatible online service.

## Connection to your signed and directed GNN research

The shipped stress study measures frozen-model response to edge deletion and causal rewiring: raw relation-adjacency displacement, embedding drift, score drift, threshold flips and predictive metrics. It separates operator change from model response. It does not claim a new bound or retraining stability result.

For a later signed/magnetic model, define the sign and direction from facts available at the scoring time. Fraud/legitimate labels must not be converted into edge signs. Time direction is not interchangeable with the direction of a monetary transfer. Reversing a time edge that introduces future information is invalid in the online protocol.

If a defensible signed transaction network becomes available, compare charge settings such as q = 0, 1/8 and 1/4 with appropriate operator normalization, a conventional relational baseline and a matched retraining study. Treat your earlier mathematical findings as hypotheses to examine in this new setting, not as automatically transferable fraud guarantees.
