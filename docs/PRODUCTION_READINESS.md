# Production readiness and acceptance gates

**Release classification: tested local reference. Production approval: no.**

The code demonstrates production engineering practices and supports a complete local workflow. Production readiness is a property of a deployed system, its data, operators and failure behavior. It cannot be established from a synthetic experiment alone.

| Area | Shipped and locally exercised | Gate before real deployment |
|---|---|---|
| Model quality | Temporal testing, calibration, cost policy, uncertainty, multiple simulation seeds | Authorized real data, agreed outcome definition, independent final period, intervention evaluation |
| Data contract | Bounds, categories, unique IDs, finite values, unknown-field rejection | Source-system contracts, schema evolution, data classification and feature-availability audit |
| Feature consistency | Shared offline/online feature logic with parity tests | Production feature store, partitions, late-arrival handling, backfills, TTL and retention |
| State and retries | SQLite transactions, persistence, exact retry cache, conflict detection | High availability, failover, disaster recovery, throughput and saturation tests |
| Access | Required API key and loopback default | TLS, managed identity, RBAC, least privilege, key rotation, gateway rate/body limits |
| Audit | Hash-linked scoring records and tamper test | Signed provenance, externally anchored chain heads, immutable audit export, feedback audit |
| Monitoring | Drift reports, recent review rate and processing latency, delayed-label metrics | Central metrics, alert routing, business SLIs, incident ownership and response exercises |
| Model governance | Manifest, checksum, seed and feature version | Controlled promotion registry, approvals, signed artifact provenance and rollback rehearsal |
| Review operations | Review priority and evidence brief, no blocking | Staffed queue, hard capacity management, SLA, case ownership, appeal/correction workflow |
| Packaging | Pinned dependencies, Docker recipe and GitHub Actions workflow | Container execution, image scanning, digest pins, CI execution in the target repository |
| AI assistant | Deterministic evidence and policy references | Separately evaluated LLM only if useful, with redaction, grounding, output schema and human control |
| GNN research | Offline temporal relation models, causal teacher scoring, structural stress and checkpoint reload | Real graph semantics, prospective holdout, graph-state serving, history warm-up, monitoring and promotion |
| Legal and privacy | Synthetic examples, no real customer files | Dataset rights, privacy and security review, applicable payment and data-protection obligations |

## Suggested operational acceptance targets

Set targets with the real business and deployment team. These examples are proposals, not measured service promises:

- At an agreed workload, p95 and p99 end-to-end latency stay below agreed limits and errors remain within an agreed error budget.
- Feature freshness and late-arrival rates remain within the declared contract. Failed lookups trigger the approved fallback rather than a silent default.
- Mature-label coverage is reported. Cases selected for review are not treated as an unbiased sample of all transactions.
- Review demand fits staffed capacity. Fixed-threshold breaches trigger operations review, not uncontrolled queue growth.
- Signed model rollback and database recovery are rehearsed. A model-only rollback does not erase feature history or duplicate old decisions.
- Score calibration, precision, recall, value capture and costs are measured in shadow mode before any intervention.

## Failure scenarios already tested

Duplicate retry, conflicting retry, out-of-order event, invalid/nonfinite input, unknown fields, missing key, wrong key, restart with retained history, concurrent identical submissions, model-file corruption, audit modification, future-feature leakage, same-timestamp exclusion, missing/duplicate identity joins, and feedback maturity.

## Outstanding engineering work

There is no distributed serving benchmark, external database failover test, chaos test, real incident drill, independent penetration test, dependency vulnerability audit, or real-data fairness analysis. Docker and GitHub Actions definitions are included, but their target environments were unavailable during this build. No hosted deployment or cloud infrastructure was created.

