# Phase 14 Implementation Log

## 2026-05-20 - Evaluation and inference architecture split

Phase 14 separates evaluation and inference responsibilities into dedicated
packages while preserving the existing public CLI entry points.

## Implementation scope

- `src/wafer_repro/evaluation/registry.py`
  - Added `EVALUATOR_REGISTRY`.
  - Added `create_evaluator`.

- `src/wafer_repro/evaluation/classification.py`
  - Added registered `ClassificationEvaluator`.
  - Moved classification probability prediction, prediction CSV saving,
    classification report saving, confusion matrix saving, and summary metric
    generation out of `metrics.py`.

- `src/wafer_repro/metrics.py`
  - Converted to a compatibility facade that re-exports the classification
    evaluation helpers.

- `src/wafer_repro/inference/inputs.py`
  - Added `InferenceInput`.
  - Added image-folder input loading.
  - Added WM-811K row/original-index/npy input loading.

- `src/wafer_repro/evaluate.py`
  - Now uses `create_evaluator("classification")`.
  - CLI behavior remains compatible.

- `src/wafer_repro/infer.py`
  - Now delegates modality-specific input loading to `inference.inputs`.
  - CLI behavior remains compatible.

- `src/wafer_repro/experiment/runner.py`
  - Test evaluation now goes through the evaluator registry.

## Verification

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\evaluation\__init__.py `
  src\wafer_repro\evaluation\registry.py `
  src\wafer_repro\evaluation\classification.py `
  src\wafer_repro\inference\__init__.py `
  src\wafer_repro\inference\inputs.py `
  src\wafer_repro\metrics.py `
  src\wafer_repro\evaluate.py `
  src\wafer_repro\infer.py `
  src\wafer_repro\experiment\runner.py
```

Result: passed.

Runner regression:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase14_eval_registry
```

Result: completed train and test evaluation through the evaluator registry.

WM-811K evaluate/infer regression:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.evaluate `
  --data data\toy_LSWMD.pkl `
  --checkpoint outputs\experiments\smoke_phase14_eval_registry\best.pt `
  --split test `
  --device cpu `
  --out-dir outputs\experiments\smoke_phase14_eval_registry\metrics_recheck

conda run -n wm811k python -m wafer_repro.infer `
  --data data\toy_LSWMD.pkl `
  --checkpoint outputs\experiments\smoke_phase14_eval_registry\best.pt `
  --row-index 0 `
  --device cpu `
  --top-k 3 `
  --out-dir outputs\experiments\smoke_phase14_eval_registry\single_infer
```

Result: both passed.

Image-folder evaluate/infer regression:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.evaluate `
  --checkpoint outputs\experiments\image_folder_phase13_artifacts\best.pt `
  --split test `
  --device cpu `
  --out-dir outputs\experiments\image_folder_phase13_artifacts\metrics_phase14_recheck

conda run -n wm811k python -m wafer_repro.infer `
  --checkpoint outputs\experiments\image_folder_phase13_artifacts\best.pt `
  --image data\toy_images\horizontal\000.png `
  --device cpu `
  --top-k 3 `
  --out-dir outputs\experiments\image_folder_phase13_artifacts\single_infer_phase14
```

Result: both passed. The image-folder inference top-1 label was `horizontal`.

## Remaining work

- Add evaluator implementations for regression, forecasting, and anomaly detection after those tasks exist.
- Move CLI wrappers into a future `wafer_repro.cli` package while preserving console entry points.
- Add batch inference outputs for directory or CSV inputs.
