# Primary sources and design attribution

Recovered references checked against primary arXiv records on 28 October 2026. Library documentation can change. Executed dependencies are recorded separately in the lock files.

1. **IEEE-CIS Fraud Detection, official competition.** https://www.kaggle.com/competitions/ieee-fraud-detection
   The benchmark task uses Vesta e-commerce transaction data.
2. **Official data description.** https://www.kaggle.com/competitions/ieee-fraud-detection/data
   Transaction and identity tables join on `TransactionID`; identity is not present for every transaction.
3. **Official evaluation.** https://www.kaggle.com/competitions/ieee-fraud-detection/overview/evaluation
   Competition submissions use ROC-AUC. This project's temporal holdout is a separate evaluation design.
4. **scikit-learn, common pitfalls and recommended practices.** https://scikit-learn.org/stable/common_pitfalls.html
   Train/test separation and fitting preprocessing only on training data inform the pipelines here.
5. **scikit-learn, precision-recall.** https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
   Average precision and precision-recall analysis guide the rare-event evaluation.
6. **scikit-learn, probability calibration.** https://scikit-learn.org/stable/modules/calibration.html
   Held-out calibration and reliability diagrams inform probability evaluation. Brier loss also depends on discrimination and outcome uncertainty, so it is not a pure calibration measure.
7. **scikit-learn, model persistence.** https://scikit-learn.org/stable/model_persistence.html
   Pickle-based artifacts require a trusted source and compatible environments. Hash checking alone does not authenticate a model.
8. **FastAPI, testing and lifespan testing.** https://fastapi.tiangolo.com/tutorial/testing/ and https://fastapi.tiangolo.com/advanced/testing-events/
   The test suite exercises application startup and requests through TestClient.

## Original project choices

The synthetic scenarios, prevalence, 120-day horizon, 55/15/10/20 elapsed-time split, two-day label delay, 5% review target, cost formula parameters, drift threshold and three seeds are project assumptions. They are not attributed to the competition organizers or claimed as industry standards. The repository contains no copied competition solution or real benchmark dataset.

## GNN extension: primary research, checked 28 October 2026

9. **Rossi et al., Temporal Graph Networks for Deep Learning on Dynamic Graphs (2020).** https://arxiv.org/abs/2006.10637
   Foundation for event-based temporal graph modeling and learned memory. Persistent TGN memory is not implemented here.
10. **Hu et al., Heterogeneous Graph Transformer (WWW 2020).** https://arxiv.org/abs/2003.01332
   Motivation for relation-specific attention and relative-time representations. The compact projected transaction model here is not a full HGT reproduction.
11. **Lu et al., BRIGHT: Graph Neural Networks in Real-Time Fraud Detection (CIKM 2022).** https://arxiv.org/abs/2205.13084
   Motivation for historical-only graph flow and separating graph computation from latency-sensitive inference. This project does not reproduce its production architecture or latency claims.
12. **Hari, Sahayam and Srinivasavarma, RGH-GNN: Risk-Gated Heterogeneous Graph Neural Network for Financial Fraud Detection (preprint, 19 October 2026).** https://arxiv.org/abs/2610.27107
   Recent motivation for risk-guided graph messages. Its stratified experiment differs from our temporal protocol. The original model here uses a past-only logistic teacher and relative-time attention rather than reproducing its architecture.
13. **Yan, Temporal Heterogeneous Graph Transformer for Credit Card Fraud Detection (preprint, 7 September 2026).** https://arxiv.org/abs/2609.07100
   Reviewed as a recent comparator. Its within-transaction relation tokens do not establish historical entity-neighborhood aggregation. Naming a model a graph transformer is not sufficient evidence that it learns from other transactions.
14. **PyTorch, macOS x86 builds deprecation announcement.** https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690
   The Linux CPU add-on is recommended for the user's Intel Mac workflow because modern official macOS x86 wheels are no longer supplied. No Docker or remote execution is claimed as completed.

All implementation choices, experiment results and robustness quantities are original to this reference unless explicitly attributed. No literature metric is presented as a reproduced score. Recent preprints are research leads, not independent proof of production suitability.

