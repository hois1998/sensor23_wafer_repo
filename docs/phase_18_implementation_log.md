# Phase 18 Implementation Log

## 2026-05-20 - Time-series modality smoke path

Phase 18 adds the first non-image modality: a small time-series classification
path. It reuses the existing classification task, supervised trainer, evaluator,
and artifact contract.

## Implementation scope

- `src/wafer_repro/datasets/timeseries/datamodule.py`
  - Added `timeseries_window` DataModule.
  - Reads wide CSV time-series features with configurable feature prefix.
  - Produces `DataBundle` with train/val/test records and datasets.

- `src/wafer_repro/models.py`
  - Added `TimeSeriesCNN`.
  - Registered `timeseries_cnn`.

- `src/wafer_repro/datasets/registry.py`
  - Registers the time-series data module.

- `src/wafer_repro/core/validation.py`
  - Adds path and split-strategy validation for `timeseries_window`.

- `src/wafer_repro/evaluate.py`
  - Adds independent evaluation support for `timeseries_window` checkpoints.

- `scripts/make_toy_timeseries.py`
  - Generates a small three-class synthetic time-series CSV dataset.

- `configs/experiments/timeseries/000_smoke.yaml`
  - Adds a smoke experiment for `timeseries_window` + `timeseries_cnn`.

## Verification

Toy dataset:

```powershell
python scripts\make_toy_timeseries.py `
  --out data\toy_timeseries.csv `
  --per-class 24 `
  --length 64
```

Result: wrote 72 rows.

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\datasets\timeseries\datamodule.py `
  src\wafer_repro\datasets\registry.py `
  src\wafer_repro\models.py `
  src\wafer_repro\core\validation.py `
  scripts\make_toy_timeseries.py
```

Result: passed.

Config validation:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.validate_config `
  --config configs\experiments\timeseries\000_smoke.yaml `
  --check-paths
```

Result: returned `status: valid`.

Registry check:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -c "from wafer_repro.datasets.registry import registered_data_module_names; from wafer_repro.models import MODEL_REGISTRY; print(registered_data_module_names()); print('timeseries_cnn' in MODEL_REGISTRY.keys())"
```

Result: `timeseries_window` and `timeseries_cnn` are registered.

Smoke train:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\timeseries\000_smoke.yaml `
  --set runtime.run_name=timeseries_phase18_smoke
```

Result: completed 2 epochs, checkpoint save, artifact generation, and test
evaluation.

Independent evaluation:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.evaluate `
  --data data\toy_timeseries.csv `
  --checkpoint outputs\experiments\timeseries_phase18_smoke\best.pt `
  --split test `
  --device cpu `
  --out-dir outputs\experiments\timeseries_phase18_smoke\metrics_recheck
```

Result: passed.

Collector regression:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments `
  --out outputs\experiments\comparison_phase18.csv
```

Result: includes `timeseries_phase18_smoke` with model `timeseries_cnn`.

## Remaining work

- Add forecasting task and sequence-to-value regression in a later phase.
- Add time-aware split strategies that prevent temporal leakage.
- Add scaler fit-on-train preprocessing artifacts.
