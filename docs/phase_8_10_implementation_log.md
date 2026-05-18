# Phase 8-10 Implementation Log

## 2026-05-18 - Phase 8 DataModule registry

Phase 8 converted dataset construction into a registry-based extension point.
`ExperimentRunner` no longer owns dataset-specific branching for `wm811k` and
`image_folder`; each data module now returns a shared `DataBundle` contract.

### Implementation scope

- `src/wafer_repro/datasets/base.py`
  - Added `DataBundle`, the common payload for labels, split records, datasets,
    data summary, resolved split strategy, and module metadata.
- `src/wafer_repro/datasets/registry.py`
  - Added `DATA_MODULE_REGISTRY` and `create_data_bundle(module_name, config)`.
  - Built-in data modules self-register through import side effects.
- `src/wafer_repro/datasets/wm811k/datamodule.py`
  - Added registered `build_wm811k_bundle`.
  - Moved WM-811K dataset construction and summary generation out of the runner.
- `src/wafer_repro/datasets/image_folder/datamodule.py`
  - Replaced the local bundle dataclass with the shared `DataBundle`.
  - Registered `build_image_folder_bundle` as `image_folder`.
- `src/wafer_repro/experiment/runner.py`
  - Replaced hard-coded dataset branching with `create_data_bundle`.

### Verification

```powershell
python -m py_compile `
  src\wafer_repro\datasets\base.py `
  src\wafer_repro\datasets\registry.py `
  src\wafer_repro\datasets\wm811k\datamodule.py `
  src\wafer_repro\datasets\image_folder\datamodule.py `
  src\wafer_repro\experiment\runner.py
```

Result: passed.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase8_wm
```

Result: completed training and test evaluation through the registry path.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\image_folder\000_smoke.yaml `
  --set runtime.run_name=image_folder_phase8
```

Result: completed training and test evaluation through the same runner path.

## 2026-05-18 - Phase 9 Scheduler and early stopping controls

Phase 9 added train-loop controls that can be varied from YAML or CLI
overrides: learning-rate schedulers, configurable checkpoint monitor selection,
and early stopping.

### Implementation scope

- `src/wafer_repro/training/callbacks.py`
  - Added monitor key resolution for names such as `val/macro_f1` and `val/loss`.
  - Added `EarlyStopping`.
  - Added scheduler factories for `none`, `step_lr`, `cosine_annealing`, and
    `reduce_on_plateau`.
- `src/wafer_repro/train.py`
  - Added CLI/config mappings for `train.scheduler.*`.
  - Added CLI/config mappings for `train.early_stopping.*`.
- `src/wafer_repro/experiment/runner.py`
  - Added scheduler creation and stepping after validation.
  - Added checkpoint monitor selection through `train.checkpoint.monitor`.
  - Added early stopping state into `run_manifest.json`.
  - Added `lr` to `history.csv`.
- `configs/experiments/wm811k/000_smoke.yaml`
  - Added explicit scheduler and early-stopping defaults.
- `configs/experiments/image_folder/000_smoke.yaml`
  - Added explicit scheduler and early-stopping defaults.

### Verification

```powershell
python -m py_compile `
  src\wafer_repro\training\callbacks.py `
  src\wafer_repro\experiment\runner.py `
  src\wafer_repro\train.py
```

Result: passed.

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.validate_config --config configs\experiments\wm811k\000_smoke.yaml
python -m wafer_repro.validate_config --config configs\experiments\image_folder\000_smoke.yaml
```

Result: both configs valid.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase9_es `
  --set train.max_epochs=3 `
  --set train.scheduler.name=step_lr `
  --set train.scheduler.step_size=1 `
  --set train.scheduler.gamma=0.5 `
  --set train.early_stopping.enabled=true `
  --set train.early_stopping.patience=0 `
  --skip-test
```

Result: completed 3 epochs, and `history.csv` recorded lr values
`0.0001`, `5e-05`, and `2.5e-05`.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase9_stop_forced `
  --set train.max_epochs=3 `
  --set train.early_stopping.enabled=true `
  --set train.early_stopping.monitor=val/loss `
  --set train.early_stopping.mode=min `
  --set train.early_stopping.patience=0 `
  --set train.early_stopping.min_delta=10.0 `
  --skip-test
```

Result: stopped at epoch 2 and wrote `stopped_early: true` plus
`trained_epochs: 2` to `run_manifest.json`.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\image_folder\000_smoke.yaml `
  --set runtime.run_name=image_folder_phase9
```

Result: completed train, checkpoint save, and test evaluation with default
scheduler and early-stopping settings.
