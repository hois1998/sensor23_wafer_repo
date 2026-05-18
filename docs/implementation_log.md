# Implementation Log

## 2026-05-18 - Phase 1 YAML experiment config foundation

사양서 `docs/experiment_platform_spec.md`의 단계별 전환 계획 중 Phase 1을 구현했다. 이번 변경의 목표는 기존 CLI 기반 학습 경로를 유지하면서, 동일 실험을 YAML configuration으로 정의하고 실행 결과에 resolved config와 실행 환경 정보를 artifact로 남기는 것이다.

### 구현 범위

- `src/wafer_repro/core/config.py`
  - YAML load 지원
  - `base_config` merge 지원
  - `--set key=value` 형태 override 지원
  - dotted path get/set helper 추가
  - resolved config hash 생성
  - `fixed.controls` 검증 기능 추가

- `src/wafer_repro/core/environment.py`
  - Python, OS, dependency version, torch, Git 상태를 `environment.json`으로 저장할 수 있게 했다.

- `src/wafer_repro/validate_config.py`
  - YAML config를 실행 전 검증하고 config hash를 출력하는 CLI module을 추가했다.

- `src/wafer_repro/train.py`
  - `--config` 옵션 추가
  - `--set` 옵션 추가
  - YAML 값이 argparse 기본값으로 들어가고, 명령줄 인자가 그 위를 덮는 구조를 추가했다.
  - 실행 directory에 다음 artifact를 저장한다.
    - `source_config.yaml`
    - `resolved_config.yaml`
    - `config_hash.txt`
    - `environment.json`
  - 기존 `config.json`, split CSV, history, checkpoint, metric artifact 저장은 유지했다.
  - `fixed.controls`는 YAML override뿐 아니라 CLI override 이후 resolved config에도 다시 검증한다.

- `configs/experiments/wm811k/001_paper_reproduction.yaml`
  - WM-811K 논문 재현 baseline config를 추가했다.

- `configs/experiments/wm811k/000_smoke.yaml`
  - toy LSWMD 데이터로 빠르게 실행 가능한 smoke config를 추가했다.

- `pyproject.toml`, `requirements.txt`
  - `PyYAML>=6.0` 의존성을 추가했다.
  - `wafer-validate-config` entry point를 추가했다.

### 검증 결과

다음 검증을 수행했다.

```powershell
python -m py_compile `
  src\wafer_repro\core\config.py `
  src\wafer_repro\core\environment.py `
  src\wafer_repro\validate_config.py `
  src\wafer_repro\train.py
```

결과: 성공.

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.validate_config --config configs\experiments\wm811k\001_paper_reproduction.yaml
python -m wafer_repro.validate_config --config configs\experiments\wm811k\000_smoke.yaml
```

결과: 두 config 모두 `status: valid`.

base Python 환경에는 `torch`가 없어 실제 학습 실행이 불가능했다. 대신 `wm811k` conda 환경의 CPU torch를 사용해 smoke 학습을 실행했다.

```powershell
$env:PYTHONPATH='src'
python .\scripts\make_toy_lswmd.py --out data\toy_LSWMD.pkl --per-class 12
conda run -n wm811k python -m wafer_repro.train --config configs\experiments\wm811k\000_smoke.yaml
```

결과: 1 epoch 학습, best checkpoint 저장, test 평가까지 성공.

생성 확인 artifact:

- `outputs/experiments/smoke_config/resolved_config.yaml`
- `outputs/experiments/smoke_config/source_config.yaml`
- `outputs/experiments/smoke_config/config_hash.txt`
- `outputs/experiments/smoke_config/environment.json`
- `outputs/experiments/smoke_config/splits/*.csv`
- `outputs/experiments/smoke_config/history.csv`
- `outputs/experiments/smoke_config/best.pt`
- `outputs/experiments/smoke_config/test_summary.json`
- `outputs/experiments/smoke_config/metrics/test_summary.json`

Smoke metric은 toy data와 1 epoch 학습이라 성능 의미는 없고, 실행 경로 검증 용도이다.

`fixed.controls`가 CLI override 이후에도 적용되는지 확인했다.

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\001_paper_reproduction.yaml `
  --seed 123 `
  --skip-test
```

결과: `data.split.seed`가 fixed value `42`와 달라져 실행 전에 오류가 발생했다. 이는 의도한 동작이다.

### 다음 단계

다음 구현 단계는 Phase 2이다.

- `data.py`를 WM-811K DataModule 구조로 분리
- raw source loading, label normalization, split, augmentation, dataset 생성을 파일별 책임으로 나누기
- 기존 CLI 호환 wrapper 유지
- external split file과 predefined split 재사용 기능 추가

## 2026-05-18 - Phase 2 WM-811K data module split

사양서 Phase 2에 맞춰 WM-811K 데이터 관련 책임을 dataset-specific package로 분리했다. 기존 `wafer_repro.data` import 경로는 유지해서 `train`, `evaluate`, `infer`가 깨지지 않도록 했다.

### 구현 범위

- `src/wafer_repro/datasets/wm811k/source.py`
  - LSWMD pickle 로드
  - wafer/failure column 탐색
  - nested label scalarization
  - failure label normalization

- `src/wafer_repro/datasets/wm811k/records.py`
  - base record 생성
  - class별 smoke sampling
  - defect class oversampling record augmentation
  - record CSV 저장/로드
  - record에 `original_index`, `labeled_index`를 보존해 external split 기준으로 사용할 수 있게 했다.

- `src/wafer_repro/datasets/wm811k/split.py`
  - stratified holdout split
  - paper-style holdout test + stratified k-fold split
  - `predefined_files` split
  - `external_test_with_train_val_split` split
  - split 간 ID 중복 검증

- `src/wafer_repro/datasets/wm811k/transforms.py`
  - wafer map to RGB 변환
  - train/eval torchvision transform 생성

- `src/wafer_repro/datasets/wm811k/dataset.py`
  - `WaferMapDataset`
  - single wafer inference tensor 생성

- `src/wafer_repro/datasets/wm811k/datamodule.py`
  - config 기반 WM-811K data module 초안
  - raw load, split, train augmentation orchestration 제공

- `src/wafer_repro/data.py`
  - 기존 공개 API를 유지하는 compatibility facade로 전환

- `src/wafer_repro/train.py`
  - YAML/CLI의 `data.split.strategy`를 실제 split 선택에 연결
  - 지원 전략:
    - `stratified_holdout`
    - `stratified_kfold`
    - `predefined_files`
    - `external_test_with_train_val_split`
  - predefined split CLI:
    - `--train-split-file`
    - `--val-split-file`
    - `--test-split-file`
  - external test split CLI:
    - `--external-test-path`
    - `--external-id-column`

### 검증 결과

문법 검증:

```powershell
python -m py_compile `
  src\wafer_repro\data.py `
  src\wafer_repro\datasets\wm811k\source.py `
  src\wafer_repro\datasets\wm811k\records.py `
  src\wafer_repro\datasets\wm811k\split.py `
  src\wafer_repro\datasets\wm811k\transforms.py `
  src\wafer_repro\datasets\wm811k\dataset.py `
  src\wafer_repro\datasets\wm811k\datamodule.py `
  src\wafer_repro\train.py
```

결과: 성공.

기존 smoke config 학습:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train --config configs\experiments\wm811k\000_smoke.yaml
```

결과: 1 epoch 학습, checkpoint 저장, test 평가 성공.

predefined split 재사용:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set data.split.strategy=predefined_files `
  --set data.split.files.train=outputs/experiments/smoke_config/splits/train_base.csv `
  --set data.split.files.val=outputs/experiments/smoke_config/splits/val.csv `
  --set data.split.files.test=outputs/experiments/smoke_config/splits/test.csv `
  --set runtime.run_name=smoke_predefined
```

결과: split strategy가 `predefined_files`로 기록되고, 기존 split CSV를 재사용해 학습과 평가 성공.

external test split:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set data.split.strategy=external_test_with_train_val_split `
  --set data.split.external_test.path=outputs/experiments/smoke_config/splits/test.csv `
  --set data.split.external_test.id_column=original_index `
  --set runtime.run_name=smoke_external_test
```

결과: external test IDs를 고정 test set으로 사용하고, 나머지에서 train/val을 재분할해 학습과 평가 성공.

기존 facade 경로 검증:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.evaluate `
  --data data\toy_LSWMD.pkl `
  --checkpoint outputs\experiments\smoke_config\best.pt `
  --split test `
  --device cpu `
  --out-dir outputs\experiments\smoke_config\metrics_recheck

conda run -n wm811k python -m wafer_repro.infer `
  --data data\toy_LSWMD.pkl `
  --checkpoint outputs\experiments\smoke_config\best.pt `
  --row-index 0 `
  --device cpu `
  --top-k 3
```

결과: `evaluate`, `infer` 모두 성공.

### 다음 단계

다음 구현 단계는 Phase 3이다.

- model registry 도입
- task registry 또는 classification task 분리
- trainer를 `training/supervised.py`로 이동
- 기존 `models.create_model`과 `train.py`는 compatibility layer로 유지

## 2026-05-18 - Phase 3 model/task/trainer registry foundation

사양서 Phase 3에 맞춰 모델 생성, task 정의, supervised training loop를 분리했다. 아직 `train.py`가 orchestration을 맡고 있지만, 핵심 학습 구성 요소는 registry 기반으로 이동했다.

### 구현 범위

- `src/wafer_repro/core/registry.py`
  - 공통 `Registry` 유틸리티 추가
  - 이름 기반 component lookup과 중복 등록 검사를 제공한다.

- `src/wafer_repro/models.py`
  - `MODEL_REGISTRY` 추가
  - 기존 `create_model()`은 유지하되 내부에서 registry builder를 호출하도록 변경
  - 기존 모델 이름 호환 유지:
    - `resnet18`
    - `efficientnet_v2_s`
    - `shufflenet_v2_x1_0`
    - `shufflenet_v2_x0_5`
    - `mobilenet_v2`
    - `mobilenet_v3_small`
    - `cnn_wdi`
    - `small_cnn`

- `src/wafer_repro/tasks/registry.py`
  - `TASK_REGISTRY`
  - `create_task()`

- `src/wafer_repro/tasks/classification.py`
  - `ClassificationTask`
  - class-weighted cross entropy 생성 책임 이동
  - epoch metric 요약 책임 이동

- `src/wafer_repro/training/registry.py`
  - `TRAINER_REGISTRY`
  - `create_trainer()`

- `src/wafer_repro/training/supervised.py`
  - `SupervisedTorchTrainer`
  - `run_epoch`
  - `cpu_state_dict`
  - `make_loader`
  - `build_optimizer`
  - optimizer 이름 지원:
    - `adam`
    - `adamw`
    - `sgd`

- `src/wafer_repro/train.py`
  - 직접 갖고 있던 loss 생성, epoch loop, loader helper를 새 모듈 호출로 전환
  - `--task-type`
  - `--trainer`
  - `--optimizer`
  - YAML의 `task.type`, `train.trainer`, `train.optimizer.name`이 실제 실행에 반영되도록 연결

### 검증 결과

문법 검증:

```powershell
python -m py_compile `
  src\wafer_repro\core\registry.py `
  src\wafer_repro\models.py `
  src\wafer_repro\tasks\registry.py `
  src\wafer_repro\tasks\classification.py `
  src\wafer_repro\training\registry.py `
  src\wafer_repro\training\supervised.py `
  src\wafer_repro\train.py
```

결과: 성공.

registry 생성 검증:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -c "from wafer_repro.models import create_model, MODEL_REGISTRY; print(MODEL_REGISTRY.keys()); m=create_model('small_cnn', num_classes=9); print(type(m).__name__)"
conda run -n wm811k python -c "from wafer_repro.tasks.registry import create_task; from wafer_repro.training.registry import create_trainer; t=create_task('classification'); tr=create_trainer('supervised_torch', task=t, device='cpu', use_amp=False); print(type(t).__name__, type(tr).__name__)"
```

결과: `small_cnn` 모델 생성, `ClassificationTask`, `SupervisedTorchTrainer` 생성 성공.

smoke 학습 검증:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase3
```

결과: 1 epoch 학습, checkpoint 저장, test 평가 성공. 기존 smoke metric과 동일한 실행 흐름을 유지했다.

### 다음 단계

다음 구현 단계는 Phase 4이다.

- `ExperimentRunner` 도입
- `train.py`의 orchestration을 runner로 이동
- run manifest 저장
- train/evaluate/infer 흐름을 runner 중심으로 정리

## 2026-05-18 - Phase 4 ExperimentRunner and run manifest

사양서 Phase 4에 맞춰 학습 실행 orchestration을 `train.py`에서 `ExperimentRunner`로 이동했다. 이제 `train.py`는 CLI argument parsing과 config 기본값 merge만 담당하고, 실제 run directory 생성, split, dataset, model/task/trainer 조립, 학습, 평가, artifact 저장은 runner가 수행한다.

### 구현 범위

- `src/wafer_repro/experiment/manifest.py`
  - timezone-aware ISO timestamp helper 추가
  - `run_manifest.json` 저장 helper 추가

- `src/wafer_repro/experiment/runner.py`
  - `ExperimentRunner`
  - `build_resolved_config`
  - `select_splits`
  - run directory와 artifact directory 생성
  - fixed control 검증
  - split 생성 및 저장
  - dataset/dataloader 생성
  - model/task/trainer/optimizer 조립
  - train/validation loop 실행
  - best/last checkpoint 저장
  - test 평가 실행
  - success/failure 상태를 `run_manifest.json`에 기록

- `src/wafer_repro/train.py`
  - 기존 orchestration 제거
  - CLI parser 유지
  - `ExperimentRunner(args, loaded_config).run()` 호출로 단순화

### run manifest schema

현재 `run_manifest.json`에는 다음 정보가 기록된다.

- `experiment_name`
- `suite`
- `run_name`
- `run_dir`
- `status`: `running`, `completed`, `failed`
- `started_at`
- `finished_at`
- `config_hash`
- `primary_metric`
- `split_strategy`
- `best_epoch`
- `best_checkpoint`
- `best_val_macro_f1`
- `test_summary`
- `error`

### 검증 결과

문법 검증:

```powershell
python -m py_compile `
  src\wafer_repro\experiment\manifest.py `
  src\wafer_repro\experiment\runner.py `
  src\wafer_repro\train.py
```

결과: 성공.

smoke 학습 검증:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\000_smoke.yaml `
  --set runtime.run_name=smoke_phase4
```

결과: 1 epoch 학습, checkpoint 저장, test 평가 성공. `outputs/experiments/smoke_phase4/run_manifest.json`에 `status: completed`, `best_epoch: 1`, `test_summary`가 기록됨.

failure manifest 검증:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\wm811k\001_paper_reproduction.yaml `
  --set runtime.run_name=fixed_fail_phase4 `
  --seed 123 `
  --skip-test
```

결과: fixed control 위반으로 실행 실패. `outputs/experiments/fixed_fail_phase4/run_manifest.json`에 `status: failed`와 오류 메시지가 기록됨.

### Commit

Phase 1-3 기준 커밋:

- `5fa9de4 Add YAML experiment foundation and modular training components`

Phase 4는 별도 커밋으로 마무리한다.

## 2026-05-18 - Phase 5 Sweep runner and comparison artifacts

사양서 Phase 5에 맞춰 YAML 기반 sweep 실행과 suite 결과 비교 기능을 추가했다. 기존 `run_experiments.py`는 paper reproduction 전용 runner로 유지하고, 새 sweep runner는 grid/manual 실험 확장을 담당한다.

### 구현 범위

- `src/wafer_repro/experiment/sweep.py`
  - sweep YAML load
  - base experiment config load
  - fixed values 적용
  - fixed values를 `fixed.controls`에 반영
  - grid expansion
  - manual expansion
  - seed/fold repeats 확장
  - trial config materialization
  - sequential trial execution
  - `sweep_manifest.json` 저장
  - `sweep_status.json` 저장

- `src/wafer_repro/sweep.py`
  - `python -m wafer_repro.sweep --config ...`
  - `--dry-run` 지원

- `src/wafer_repro/analysis/collector.py`
  - run directory에서 `test_summary.json`, `run_manifest.json`, `config.json`, `data_summary.json` 수집
  - `sweep_manifest.json`의 axes metadata를 읽어 `axis_*` 컬럼 추가
  - `comparison_trials.csv` 생성
  - `comparison_trials_grouped.csv` 생성

- `src/wafer_repro/collect_results.py`
  - 기존 CLI를 새 collector 구현으로 연결

- `configs/sweeps/wm811k_smoke_grid.yaml`
  - smoke용 preprocessing axis grid sweep 추가
  - `colormap` vs `replicate` 비교

- `pyproject.toml`
  - `wafer-sweep = "wafer_repro.sweep:main"` entry point 추가

### 검증 결과

문법 검증:

```powershell
python -m py_compile `
  src\wafer_repro\experiment\sweep.py `
  src\wafer_repro\sweep.py `
  src\wafer_repro\analysis\collector.py `
  src\wafer_repro\collect_results.py
```

결과: 성공.

dry-run 검증:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_grid.yaml `
  --dry-run
```

결과: `outputs/experiments/wm811k_smoke_grid/_trial_configs` 아래 trial config 2개 생성.

실제 sweep 실행:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.sweep `
  --config configs\sweeps\wm811k_smoke_grid.yaml
```

결과: `preprocessing_colormap`, `preprocessing_replicate` 두 trial 모두 학습과 test 평가 성공.

결과 수집:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments\wm811k_smoke_grid `
  --out outputs\experiments\wm811k_smoke_grid\comparison_trials.csv
```

결과:

- `comparison_trials.csv` 생성
- `comparison_trials_grouped.csv` 생성
- `axis_preprocessing` 컬럼으로 sweep axis가 기록됨

### 다음 단계

다음 구현 단계는 Phase 6이다.

- WM-811K 외 신규 modality 추가
- 우선 `image_folder` classification DataModule을 추가해 dataset plugin 구조를 검증
- smoke용 tiny image dataset 생성 스크립트 또는 config 추가
- 기존 supervised trainer를 재사용해 신규 modality 실행 가능성 검증

## 2026-05-18 - Phase 6 Image-folder modality smoke path

사양서 Phase 6에 맞춰 WM-811K 외 신규 modality를 추가했다. 첫 신규 modality는 가장 단순한 class-folder image classification이며, 기존 `ExperimentRunner`, `ClassificationTask`, `SupervisedTorchTrainer`, `small_cnn`을 그대로 재사용해 학습과 평가가 가능하도록 했다.

### 구현 범위

- `src/wafer_repro/datasets/image_folder/datamodule.py`
  - class folder scan
  - image records 생성
  - stratified holdout split
  - image transform 생성
  - `ImageFolderRecordsDataset`
  - image-folder data summary 생성

- `src/wafer_repro/experiment/runner.py`
  - `data.module: image_folder` 분기 추가
  - image-folder labels를 task class order로 반영
  - 기존 supervised training/evaluation path 재사용
  - 기존 WM-811K path 유지

- `scripts/make_toy_image_folder.py`
  - smoke 검증용 tiny image-folder dataset 생성
  - class: `diagonal`, `horizontal`, `vertical`

- `configs/experiments/image_folder/000_smoke.yaml`
  - image-folder smoke experiment config 추가

- `.gitignore`
  - 생성된 toy image data가 Git에 들어가지 않도록 `data/toy_images/` 추가

### 검증 결과

toy image dataset 생성:

```powershell
python .\scripts\make_toy_image_folder.py `
  --out data\toy_images `
  --per-class 12 `
  --size 48
```

결과: 3개 class, 총 36개 PNG 생성.

문법 및 config 검증:

```powershell
python -m py_compile `
  src\wafer_repro\datasets\image_folder\datamodule.py `
  src\wafer_repro\experiment\runner.py `
  scripts\make_toy_image_folder.py

$env:PYTHONPATH='src'
python -m wafer_repro.validate_config `
  --config configs\experiments\image_folder\000_smoke.yaml
```

결과: 성공.

image-folder smoke 학습:

```powershell
$env:PYTHONPATH='src'
conda run -n wm811k python -m wafer_repro.train `
  --config configs\experiments\image_folder\000_smoke.yaml
```

결과: 1 epoch 학습, checkpoint 저장, test 평가 성공.

확인된 artifact:

- `outputs/experiments/image_folder_smoke/run_manifest.json`
- `outputs/experiments/image_folder_smoke/resolved_config.yaml`
- `outputs/experiments/image_folder_smoke/data_summary.json`
- `outputs/experiments/image_folder_smoke/test_summary.json`

`resolved_config.yaml`에는 다음 class order가 기록됨.

- `diagonal`
- `horizontal`
- `vertical`

결과 수집:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments `
  --out outputs\experiments\comparison_all_trials.csv
```

결과: image-folder run도 기존 collector에서 수집 가능함을 확인.

### 다음 단계

사양서에 명시된 Phase 0-6의 기본 이행은 완료했다. 이후부터는 확장 hardening 단계로 넘어간다.

- `evaluate.py`와 `infer.py`도 `ExperimentRunner`/DataModule registry 기반으로 일반화
- image-folder predefined split 지원
- `ExperimentRunner` 내부 data module branch를 registry로 전환
- scheduler, early stopping callback 추가
- sweep parallelism과 resume 기능 추가
