# Phase 16 Implementation Log

## 2026-05-20 - Multi-seed and k-fold suite aggregation

Phase 16 strengthens repeat execution and result aggregation for seed/fold
experiments. The sweep runner already expanded `repeats.seeds` and
`repeats.folds`; this phase adds a concrete smoke suite and summary artifacts.

## Implementation scope

- `src/wafer_repro/analysis/collector.py`
  - Added `build_suite_summary`.
  - `write_comparison` now also writes `<out>_summary.json`.
  - Summary includes run count, completed count, seed list, fold list, group keys,
    metric mean/std/min/max/count, and warnings.

- `configs/sweeps/wm811k_smoke_seed_fold.yaml`
  - Added a 2 seed x 2 fold smoke suite.
  - Uses `max_workers: 2` to exercise parallel execution with repeated trials.

## Verification

Syntax check:

```powershell
python -m py_compile src\wafer_repro\analysis\collector.py
```

Result: passed.

Config validation:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\sweeps\wm811k_smoke_seed_fold.yaml
```

Result: returned `status: valid`.

Dry-run expansion:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_seed_fold.yaml `
  --dry-run
```

Result: materialized four trial configs.

Actual seed/fold suite:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_seed_fold.yaml
```

Result: completed four trials:

- seed 42, fold 0
- seed 42, fold 1
- seed 43, fold 0
- seed 43, fold 1

Collector aggregation:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments\wm811k_smoke_seed_fold `
  --out outputs\experiments\wm811k_smoke_seed_fold\comparison.csv
```

Result: wrote:

- `comparison.csv`
- `comparison_grouped.csv`
- `comparison_summary.json`

`comparison_summary.json` recorded `run_count: 4`, `completed_count: 4`,
`seeds: [42, 43]`, and `folds: [0, 1]`.

## Remaining work

- Add paired axis analysis and leaderboard reports in Phase 17.
- Add stricter warnings once frozen split and dataset hash policies are fully
  enforced.
