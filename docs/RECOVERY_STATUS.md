# Recovery status and evidence

Recovery date: October 28, 2026 (America/Los_Angeles).

## Recovered sources

The private deployed Site source repository was recovered at commit `cfe59beffa54d7693026c5ac7b1668e5adb9971a`. It included the application, trained JavaScript model exports, parity fixtures, Python research source, notebooks and dependency locks. The existing 16-slide cloud presentation and its speaker notes were inspected. The cover subtitle was updated in the cloud deck. Portable exports preserve its content and structure, with Arial used consistently and editable native charts/tables.

The public GitHub repository initially contained only a 111-byte README on main, commit `3bdf4d8c9076e0cf3cf9cb77e9bc2a35549f1d88`. There were no other branches to reconcile at inspection.

The separate original Library ZIP was visible but its browser download controls did not produce a file. Its later publication-package changes, full Python checkpoints and raw multi-seed reports have not been recovered. This release is a documented recovery from deployed source, not a claim that the original archive was fully restored. Do not delete or replace that archive.

## Fresh verification

- Application: 12 passed, zero failed on Node 24.19.0 using the original lockfile and pnpm 11.25.0.
- TypeScript: successful exit.
- The application suite includes the saved 80-case parity fixture and 48-event Python-to-application parity replay.
- New local reviewer tour: 24 transactions, four neural comparisons, retry checks, workspace isolation, a revision-checked saved review, CSV export and 25 valid audit records. It ended with REVIEWER TOUR PASSED.
- Python: 41 passed, one module skipped, two warnings on Python 3.12.14 on the Intel Mac. PyTorch-dependent tests were skipped. Warnings concern FastAPI/httpx deprecation and Isolation Forest sample-size adjustment in a small artificial fixture.

The historical 50-test Python result was not rerun. New GNN training, notebook re-execution, Docker execution, target-scale load tests and a fresh authenticated hosted reviewer tour were not completed during recovery. Passing local handler tests does not validate the identity gateway or live D1 under load.

## Results and provenance

`applications/workbench/model-data/evidence.json` preserves the deployed three-seed model comparison, selection records and 12 seed-42 structural perturbations. The recovered summary is explicitly labeled as derived from this export. The three-seed structural chart values came from the existing presentation chart, not from recovered raw perturbation files. Its separate JSON records that limitation. Display charts use rounded literal data for Excel compatibility; source artifacts retain the recovered values.

All model performance is synthetic. Real IEEE-CIS results remain unmeasured. The schema smoke fixture does not establish real-data performance. The live active model remains logistic regression and GNN scores remain shadow comparisons.

## Privacy and publication

The personal hosting identity, Git metadata, dependency folders, caches, runtime databases and source-download ZIP were excluded. The public hosting example has no personal Site ID. No real transaction dataset is included. Historical JUnit machine paths were sanitized for publication.

Seera confirmed the existing portfolio's custom-domain configuration for hirpadata.com. The FraudScope entry was saved and observed on the public custom domain with the agreed title, synthetic results, limitations and source/demo URLs. Existing portfolio content was preserved. No demo access policy was changed.

## Remaining recovery work

Obtain the original FraudScope_Flagship_Project.zip, compare its source and outputs against this release, preserve any distinct later changes, and rerun affected checks before describing the complete original archive as reconciled. The full Linux GNN suite and original checkpoint replay still require their declared environment and artifacts.
