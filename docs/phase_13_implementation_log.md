# Phase 13 Implementation Log

## 2026-05-20 - Artifact contract standardization

Phase 13 adds explicit artifact contract files to each run. The goal is to make
run outputs easier to compare, audit, and reuse across code versions.

## Implementation scope

- `src/wafer_repro/experiment/artifacts.py`
  - Added data source identity generation.
  - Added optional file sha256 support through `runtime.hash_data_files`.
  - Added preprocessing manifest generation.
  - Added split CSV sha256 manifest generation.

- `src/wafer_repro/experiment/runner.py`
  - Writes `data_identity.json`.
  - Writes `preprocessing_manifest.json`.
  - Writes `splits/split_hashes.json`.
  - Writes `artifact_manifest.json`.
  - Adds artifact paths to `run_manifest.json`.
  - Creates a first-class `predictions/` directory.

- `src/wafer_repro/metrics.py`
  - Added `save_predictions`.
  - `save_evaluation` now writes prediction rows with true label, predicted label,
    confidence, and per-class probability columns.

- `src/wafer_repro/analysis/collector.py`
  - Adds artifact presence flags:
    - `has_data_identity`
    - `has_split_hashes`
    - `has_test_predictions`
  - Adds data path and data size columns when available.

## New run artifacts

Each new run now records:

```text
artifact_manifest.json
data_identity.json
preprocessing_manifest.json
splits/split_hashes.json
predictions/test_predictions.csv
```

Existing artifacts such as `resolved_config.yaml`, `config_hash.txt`,
`environment.json`, `history.csv`, `best.pt`, and metric summaries are still
written as before.

## Verification

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\experiment\artifacts.py `
  src\wafer_repro\metrics.py `
  src\wafer_repro\experiment\runner.py `
  src\wafer_repro\analysis\collector.py
```

Result: passed.

WM-811K smoke run:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase13_artifacts
```

Result: completed train and test evaluation. Verified:

- `outputs/experiments/smoke_phase13_artifacts/artifact_manifest.json`
- `outputs/experiments/smoke_phase13_artifacts/data_identity.json`
- `outputs/experiments/smoke_phase13_artifacts/preprocessing_manifest.json`
- `outputs/experiments/smoke_phase13_artifacts/splits/split_hashes.json`
- `outputs/experiments/smoke_phase13_artifacts/predictions/test_predictions.csv`

Image-folder smoke run:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\image_folder\000_smoke.yaml `
  --set runtime.run_name=image_folder_phase13_artifacts
```

Result: completed train and test evaluation. Verified the same artifact contract.

Collector regression:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments `
  --out outputs\experiments\comparison_phase13.csv
```

Result: completed and included the new artifact presence columns.

## Remaining artifact work

- Add dataset hash validation to preflight when hashes are declared in config.
- Add prediction schema validation once evaluator registry is introduced.
- Add optional parquet output if `pyarrow` is introduced in a later phase.
