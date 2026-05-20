# Phase 15 Implementation Log

## 2026-05-20 - Sweep execution hardening

Phase 15 expands the sweep runner beyond basic grid/manual execution. It adds
random sweep expansion, retry metadata, parallel execution, and per-trial status
manifests.

## Implementation scope

- `src/wafer_repro/experiment/sweep.py`
  - Added `expansion.mode: random`.
  - Added `expansion.num_trials` / `expansion.n_trials`.
  - Added `expansion.seed` for deterministic random sampling.
  - Added `execution.max_workers` for parallel trial execution.
  - Added `execution.retry_failed`.
  - Added per-trial manifests under `_trial_manifests/`.
  - Added attempt metadata, timestamps, return codes, axes, seed, and fold into
    sweep status records.

- `src/wafer_repro/core/validation.py`
  - Added validation support for random sweep mode and required positive
    `num_trials`.

- `configs/sweeps/wm811k_smoke_grid.yaml`
  - Added explicit `max_workers` and `retry_failed` defaults.

- `configs/sweeps/wm811k_smoke_random.yaml`
  - Added a tiny random sweep config for validation and smoke execution.

## Verification

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\experiment\sweep.py `
  src\wafer_repro\core\validation.py `
  src\wafer_repro\sweep.py
```

Result: passed.

Config validation:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\sweeps\wm811k_smoke_grid.yaml

conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\sweeps\wm811k_smoke_random.yaml
```

Result: both returned `status: valid`.

Random dry-run:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_random.yaml `
  --dry-run
```

Result: materialized two random trial configs and trial manifests.

Grid skip-completed regression:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_grid.yaml `
  --skip-completed
```

Result: skipped two existing completed grid trials.

Random parallel execution:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_random.yaml
```

Result: completed two random trials with `max_workers: 2`. `sweep_status.json`
recorded `completed: 2`, attempts, timestamps, axes, config paths, and run dirs.

## Remaining work

- Add richer retry policies if needed, such as retrying only known transient
  failures.
- Add distributed or process-pool execution only after CPU/GPU resource handling
  is formalized.
