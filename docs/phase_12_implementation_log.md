# Phase 12 Implementation Log

## 2026-05-20 - Config schema validation and preflight checks

Phase 12 implements the first broad validation layer for the full experiment
platform specification. The goal is to catch common configuration mistakes
before training or sweep execution begins.

## Implementation scope

- `src/wafer_repro/core/validation.py`
  - Added `ValidationIssue` and `ConfigValidationError`.
  - Added `validate_experiment_config`.
  - Added `validate_sweep_config`.
  - Added checks for:
    - `schema_version`
    - required top-level experiment sections
    - registered `data.module`
    - registered `model.name`
    - registered `task.type`
    - registered `train.trainer`
    - optimizer and scheduler names
    - checkpoint, scheduler, and early-stopping monitor keys
    - `min`/`max` monitor modes
    - basic numeric constraints
    - split strategy compatibility by data module
    - predefined/external split file declarations
    - optional local path existence checks
    - sweep fixed-axis conflicts

- `src/wafer_repro/validate_config.py`
  - Now validates both experiment YAML and sweep YAML.
  - Added `--check-paths`.
  - Emits structured JSON for both valid and invalid configs.
  - Returns a non-zero exit code for invalid configs without printing a Python traceback.

- `src/wafer_repro/experiment/runner.py`
  - Runs experiment preflight validation after resolving config and before data construction.

- `src/wafer_repro/experiment/sweep.py`
  - Validates sweep configs before trial expansion.

- `src/wafer_repro/datasets/registry.py`
  - Added `registered_data_module_names` for validation.

## Verification

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\core\validation.py `
  src\wafer_repro\validate_config.py `
  src\wafer_repro\experiment\runner.py `
  src\wafer_repro\experiment\sweep.py `
  src\wafer_repro\datasets\registry.py
```

Result: passed.

Valid experiment configs:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --check-paths

conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\experiments\image_folder\000_smoke.yaml `
  --check-paths
```

Result: both returned `status: valid`.

Valid sweep config:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\sweeps\wm811k_smoke_grid.yaml
```

Result: returned `status: valid`.

Invalid model name:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set model.name=missing_model
```

Result: returned `status: invalid` with a `model.name` registry error.

Invalid data path:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set data.source.path=data\missing.pkl `
  --check-paths
```

Result: returned `status: invalid` with a `data.source.path` file error.

Invalid sweep fixed-axis conflict:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\sweeps\wm811k_smoke_grid.yaml `
  --set sweep.fixed.data.preprocessing.channel_mode=colormap
```

Result: returned `status: invalid` because the `replicate` axis would override
the fixed `data.preprocessing.channel_mode` value.

Regression smoke train:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase12_preflight
```

Result: completed train, checkpoint save, and test evaluation.

Sweep resume regression:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_grid.yaml `
  --skip-completed
```

Result: skipped the two existing completed trials.

## Remaining validation work

Phase 12 establishes the validation layer, but a few spec items remain for
later phases:

- prediction artifact schema validation
- dataset hash and split hash validation
- evaluator/metric registry validation after evaluator registry exists
- richer validation for future task types such as regression, forecasting, and anomaly detection
