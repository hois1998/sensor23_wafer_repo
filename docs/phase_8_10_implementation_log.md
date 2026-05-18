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
