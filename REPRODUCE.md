# Reproduction commands

Run commands from the artifact root. Python 3.12.7, NumPy 1.26.4, scikit-learn 1.5.1, and PyTorch 2.8.0+cu126 were used for the matched-label estimator experiments. Complete per-run configuration and device information are in the saved `started.json` and `completed.json` files. The lightweight saved-results verifier needs only NumPy.

## Saved numerical evidence

```powershell
python -m pip install -r requirements-saved.txt
python tools/verify_saved_results.py
python tools/verify_saved_results.py --bootstrap
```

The verifier recomputes the saved candidate and default-gated metrics, complete measured throughput block goodput, and optionally the residual-HGB paired SQL-cluster intervals. It does not replay the database. Bootstrap intervals condition on the fitted models and selected configurations, preserve copies of an underlying SQL query together, and use 2,000 resamples.

## Fresh matched-label training

Make a separate working copy using `tools/new_estimator_replication.py`. It copies the input checkpoints/workloads, experiment code, protocols, baseline source, and original DuckDB audit while leaving the new training output directories empty.

```powershell
python tools/new_estimator_replication.py --destination ../PERSEUS-replication
cd ../PERSEUS-replication
python -m pip install -r requirements.txt
python rebuttal_20260929/reviewer_4udk/run_estimator_controls.py --drift column
python rebuttal_20260929/reviewer_4udk/run_estimator_controls.py --drift table
python rebuttal_20260929/reviewer_4udk/run_estimator_controls.py --drift value
python rebuttal_20260929/reviewer_4udk/analyze_estimator_controls.py
```

These are the recorded estimator training and analysis entry points. The new manifests identify the relocated release sources and new observations. Structural drifts use 2,000 fitting, 500 tuning, and 500 gate labels; value drift uses 500/250/250. Each estimator family evaluates four configurations using tuning labels only. Full MSCN retraining also replays source data. Three outer adaptation splits are 501/502/503.

Use CPU-compatible PyTorch or install the matching CUDA 12.6 wheel from PyTorch's official index when reproducing the GPU environment. Newly fitting models and replaying database timings can produce stochastic and machine-dependent differences.

## PostgreSQL native callback

The three callback integration files used locally are supplied under `native/postgresql-13.1-overlay/`, matching their PostgreSQL source-relative locations:

```
src/backend/optimizer/path/costsize.c
src/backend/utils/misc/guc.c
src/include/optimizer/cost.h
```

Apply this source overlay to PostgreSQL 13.1 with the End-to-End-CardEst-Benchmark integration, rebuild using PostgreSQL's Windows MSVC instructions, and run a dedicated instance. The recorded experiment used Visual C++ build 1944, a local server on port 5434, database `postgres`, and schema `stats_inj`. The overlay documents the actual source implementation; an independent clean Windows compiler/build has not been verified as part of this packaging task.

The map key is `query_id + sorted alias_set + method`; the callback replaces planner rows before plan selection. Required GUCs and map parsing are defined in the supplied `guc.c` and `costsize.c`. Database timing replay requires the original STATS database rows from the benchmark and a database-server-readable map path. Set the PostgreSQL connection password through `PGPASSWORD`, rather than placing credentials in files or command arguments.

The recorded protocol-attribution phase sequence is:

```powershell
python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py prepare
python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py validate
python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py select
python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py test
python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py analyze
python rebuttal_20260929/reviewer_4udk/run_throughput.py
```

New native runs need a separate working copy with old phase outputs removed from that copy. These commands are historical replay entry points, not an instruction to overwrite the supplied observations. Estimator repair, runtime-rule selection, and final-test replay are separate phases; freeze the selected rule before test execution.

The broader main-paper entry points are `run_workload_drift_native_optimizer_callback.py`, `summarize_native_callback_policy_grid.py`, and the SQL/hint probes in `experiments/`. Their command-line help defines the database parameters. Main-paper results and the later matched-label/native supplements retain separate manifests and populations.
