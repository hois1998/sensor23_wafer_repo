# 확장형 실험 플랫폼 전환 사양서

이 문서는 현재 `sensors23_wafer_repro` 프로젝트를 단일 논문 재현 코드에서 범용 실험 플랫폼으로 확장하기 위한 상세 설계서이다. 현재 코드는 WM-811K/LSWMD wafer map 데이터셋을 대상으로 Sensors 2023 논문 설정을 최대한 따라가는 구조에 가깝다. 앞으로의 목표는 특정 논문 세팅 하나를 재현하는 것이 아니라, 동일 데이터 또는 다른 데이터에 대해 전처리, split, 모델, 하이퍼파라미터, 학습 장치, 추론 방식, 평가 지표를 체계적으로 바꾸며 어떤 변주가 성능 개선에 기여하는지 검증할 수 있는 실험 시스템으로 발전시키는 것이다.

핵심 방향은 다음과 같다.

- 모든 실험은 YAML configuration으로 정의한다.
- YAML은 데이터, 전처리, split, 모델, 학습, 추론, 평가, 출력 경로를 한 곳에서 표현한다.
- 실험 간 공정 비교를 위해 고정해야 하는 요소를 별도 namespace로 명시한다.
- 실험 축은 `m`개의 변화 방향과 각 방향별 `n`개의 방법론으로 표현하고, 이를 grid, random, manual ablation 형태로 실행할 수 있게 한다.
- WM-811K뿐 아니라 이미지, 시계열, tabular, sensor, 강화학습 환경까지 확장 가능한 모듈 경계를 둔다.
- 지도학습, 비지도학습, 자기지도학습, 이상탐지, 강화학습을 하나의 거대한 함수로 합치지 않고 task plugin 형태로 분리한다.
- 학습, 평가, 추론 결과는 동일한 artifact 규약으로 저장해서 후처리와 비교가 자동화되도록 한다.

---

## 1. 현재 프로젝트의 성격

현재 프로젝트는 다음 성격을 가진다.

- 데이터는 기본적으로 `../LSWMD.pkl`를 읽는다.
- `wafer_repro.data`가 WM-811K 데이터 로딩, label 정규화, split, augmentation, dataset 생성을 동시에 담당한다.
- `wafer_repro.train`이 CLI argument를 직접 받고, split 생성, dataset 생성, model 생성, optimizer 생성, training loop, checkpoint 저장, test 평가까지 한 번에 수행한다.
- `wafer_repro.models`는 논문 비교 모델과 간단한 CNN을 이름 기반 factory로 생성한다.
- `wafer_repro.metrics`는 classification report, confusion matrix, summary metric 저장을 담당한다.
- `wafer_repro.evaluate`는 이미 저장된 checkpoint와 split CSV를 이용해 재평가한다.
- `wafer_repro.infer`는 단일 wafer map 또는 `.npy` 파일에 대해 추론한다.
- `wafer_repro.run_experiments`는 paper model 목록과 fold 조합을 순차 실행한다.
- `wafer_repro.collect_results`는 run directory 아래의 `test_summary.json`을 모아 CSV를 만든다.
- `wafer_repro.benchmark`는 모델별 parameter, MACs/FLOPs, throughput을 측정한다.

현재 코드는 논문 재현에는 유용하지만, 범용 실험 플랫폼으로 쓰기에는 다음 한계가 있다.

- 실험 정의가 CLI argument에 흩어져 있어 복잡한 실험군을 재현하기 어렵다.
- 데이터 로딩, split, augmentation, dataset 생성이 WM-811K에 강하게 결합되어 있다.
- 모델 factory는 CNN image classification에 맞춰져 있고 task별 head나 output contract가 분리되어 있지 않다.
- early stopping, scheduler, gradient accumulation, multi seed, nested cross validation, ensemble, threshold tuning 같은 학습 장치를 확장하려면 `train.py`가 계속 커질 가능성이 높다.
- 고정 파라미터와 변경 파라미터의 구분이 없다.
- split 파일, external validation/test set, preprocessing artifact, dataset hash, environment 정보 같은 비교 신뢰성 요소가 표준화되어 있지 않다.
- 결과 비교는 `test_summary.json` 중심이라, fold/seed/model/preprocess 축을 풍부하게 분석하기 어렵다.

이 문서의 목적은 이 한계를 해소할 새 구조를 정리하는 것이다.

---

## 2. 설계 원칙

### 2.1 Configuration is the experiment

실험은 코드 안에 숨겨진 값이 아니라 YAML 파일로 표현되어야 한다. CLI는 실험을 실행하는 진입점일 뿐이며, 실제 실험 조건은 `resolved_config.yaml`로 저장되어야 한다.

필수 규칙:

- 모든 실행은 원본 YAML과 merge가 끝난 `resolved_config.yaml`을 저장한다.
- 실행 결과 directory는 config hash와 사람이 읽을 수 있는 이름을 함께 사용한다.
- training, evaluation, inference는 동일 config를 참조한다.
- YAML에 없는 값이 코드 기본값으로 조용히 들어가는 것을 최소화한다.

### 2.2 Fixed controls must be first-class

비교 실험에서 중요한 것은 바꾸는 것보다 고정하는 것이다. 예를 들어 preprocessing 방법을 비교하려면 model, split seed, test set, training seed, epoch, optimizer, metric selection 기준이 고정되어야 한다.

따라서 YAML에는 `fixed` namespace를 둔다.

예:

```yaml
fixed:
  dataset_id: wm811k_lswmd_v1
  split:
    strategy: stratified_holdout
    seed: 42
    test_size: 0.2
    val_fraction_of_trainval: 0.25
  labels:
    order: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
  metric:
    primary: macro_f1
    selection_split: val
  training:
    max_epochs: 30
    batch_size: 128
```

`fixed`에 들어간 값은 sweep에서 덮어쓰려 할 때 경고 또는 오류를 내는 편이 좋다. 단, 연구자가 의도적으로 fixed를 바꾸고 싶을 때는 `--allow-fixed-override` 같은 명시 옵션을 요구한다.

### 2.3 Separate modality, task, algorithm, runtime

데이터 형태와 학습 문제와 알고리즘과 실행 환경은 서로 다른 축이다.

- modality: wafer map, image, time series, tabular, graph, text, RL environment
- task: classification, regression, anomaly detection, clustering, forecasting, policy learning
- algorithm: CNN, Transformer, RandomForest, AutoEncoder, PPO 등
- runtime: device, precision, num_workers, distributed, output directory

이 네 가지를 한 class에 합치면 확장성이 떨어진다. 각 축은 registry와 interface로 분리한다.

### 2.4 Artifacts are contracts

실험 결과 파일은 사람이 보기 위한 로그만이 아니라 후속 자동 분석의 입력이다. 따라서 저장 파일 이름과 schema를 표준화한다.

권장 artifact:

- `resolved_config.yaml`
- `config_hash.txt`
- `dataset_summary.json`
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- `preprocessing/preprocessor.pkl` 또는 modality별 transform manifest
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `history.csv`
- `metrics/val_summary.json`
- `metrics/test_summary.json`
- `predictions/test_predictions.parquet` 또는 CSV
- `environment.json`
- `run_manifest.json`

### 2.5 Leakage prevention by design

성능 비교에서 가장 위험한 문제는 data leakage이다. 따라서 다음 규칙을 구조적으로 강제한다.

- split은 augmentation보다 먼저 수행한다.
- train split에 fit한 preprocessing 통계만 val/test에 transform으로 적용한다.
- external test path가 지정되면 train/val 생성 과정에서 해당 sample id를 제외한다.
- k-fold에서는 fold별 train/val/test index를 명시 파일로 저장한다.
- test set은 model selection에 사용하지 않는다.

---

## 3. 용어 정의

| 용어 | 의미 |
|---|---|
| Experiment | 하나의 완결된 실험 정의. YAML 하나 또는 resolved config 하나로 표현된다. |
| Trial | Experiment를 특정 seed, fold, sweep 조합으로 실행한 단위. |
| Run | 실제 실행 instance. 재시도하면 같은 Trial이라도 Run은 여러 개일 수 있다. |
| Suite | 여러 Experiment 또는 sweep을 묶은 비교 단위. |
| Axis | 실험에서 변화시키는 방향. 예: preprocessing, model, optimizer, scheduler. |
| Method | 특정 Axis 안의 선택지. 예: `colormap`, `replicate`, `mobilenet_v3_small`. |
| Fixed Control | 비교를 위해 고정하는 설정. 예: split seed, test IDs, batch size. |
| DataModule | raw data를 읽고 split, preprocessing, dataset, dataloader를 만드는 단위. |
| Task | classification, regression, anomaly detection 등 loss, metric, prediction format을 결정하는 문제 정의. |
| ModelModule | config로부터 모델을 만들고 checkpoint load/save 계약을 제공하는 단위. |
| Trainer | 학습 loop, optimizer, scheduler, callback을 실행하는 단위. |
| Evaluator | prediction을 metric artifact로 변환하는 단위. |
| Predictor | checkpoint와 input을 받아 inference artifact를 만드는 단위. |
| Registry | 문자열 이름을 class/function으로 연결하는 plugin map. |

---

## 4. 현재 코드와 목표 구조 매핑

| 현재 파일 | 현재 책임 | 목표 구조에서의 위치 |
|---|---|---|
| `src/wafer_repro/data.py` | LSWMD 로딩, label 정규화, split, augmentation, dataset, inference tensor | `datasets/wm811k/source.py`, `datasets/wm811k/preprocess.py`, `data/splitters.py`, `datasets/wm811k/datamodule.py`, `data/transforms.py` |
| `src/wafer_repro/train.py` | CLI, config, split, dataloader, train loop, checkpoint, test 평가 | `cli/run.py`, `experiment/runner.py`, `training/supervised.py`, `callbacks/checkpoint.py`, `evaluation/classification.py` |
| `src/wafer_repro/models.py` | model name 기반 factory, paper model list | `models/registry.py`, `models/torchvision.py`, `models/wafer_cnn.py` |
| `src/wafer_repro/metrics.py` | classification metric 저장 | `evaluation/classification.py`, `evaluation/artifacts.py` |
| `src/wafer_repro/evaluate.py` | checkpoint 재평가 CLI | `cli/evaluate.py`, `experiment/evaluator.py` |
| `src/wafer_repro/infer.py` | 단일 샘플 추론 CLI | `cli/infer.py`, `inference/predictor.py`, `inference/io.py` |
| `src/wafer_repro/run_experiments.py` | paper model x fold sequential runner | `experiment/sweep.py`, `cli/sweep.py` |
| `src/wafer_repro/collect_results.py` | run summary CSV 수집 | `analysis/collector.py`, `analysis/leaderboard.py` |
| `src/wafer_repro/benchmark.py` | 모델 효율성 측정 | `benchmark/model_profile.py`, `cli/benchmark.py` |
| `src/wafer_repro/utils.py` | seed, device, json, directory | `core/random.py`, `core/device.py`, `core/io.py` |
| `src/wafer_repro/labels.py` | paper labels, alias | `datasets/wm811k/labels.py`, 또는 config labels |

---

## 5. 권장 패키지 구조

아래 구조는 이름 예시이다. 기존 package 이름을 유지하려면 `wafer_repro` 아래에 동일한 하위 폴더를 만들 수 있다. 다만 장기적으로 wafer 전용 이름보다 `experiment_lab`, `ml_lab`, `xlab` 같은 일반 이름이 더 적합할 수 있다.

```text
sensors23_wafer_repro/
  configs/
    base/
      runtime_cpu.yaml
      runtime_gpu.yaml
      classification_defaults.yaml
    datasets/
      wm811k_base.yaml
      wm811k_fixed_split.yaml
    experiments/
      wm811k/
        001_paper_reproduction.yaml
        010_preprocess_ablation.yaml
        020_model_ablation.yaml
        030_training_controls.yaml
    sweeps/
      wm811k_mxn_grid.yaml

  docs/
    paper_reproduction_notes.md
    experiment_platform_spec.md

  src/
    wafer_repro/
      core/
        config.py
        schema.py
        registry.py
        io.py
        random.py
        device.py
        hashing.py
        logging.py

      experiment/
        runner.py
        sweep.py
        trial.py
        manifest.py
        compare.py

      data/
        base.py
        records.py
        splitters.py
        transforms.py
        loaders.py
        external_sets.py

      datasets/
        wm811k/
          __init__.py
          labels.py
          source.py
          preprocess.py
          transforms.py
          datamodule.py
        image_folder/
          datamodule.py
        timeseries/
          datamodule.py

      tasks/
        base.py
        classification.py
        regression.py
        anomaly.py
        forecasting.py
        reinforcement.py

      models/
        registry.py
        torchvision_backbones.py
        wafer_cnn.py
        timeseries.py
        sklearn_models.py

      training/
        base.py
        supervised.py
        unsupervised.py
        callbacks.py
        optimizers.py
        schedulers.py
        losses.py

      evaluation/
        base.py
        classification.py
        regression.py
        anomaly.py
        aggregation.py
        artifacts.py

      inference/
        predictor.py
        io.py
        postprocess.py

      benchmark/
        model_profile.py
        throughput.py

      analysis/
        collector.py
        leaderboard.py
        report.py

      cli/
        run.py
        sweep.py
        evaluate.py
        infer.py
        compare.py
        validate_config.py
```

이 구조의 핵심은 `datasets`, `tasks`, `models`, `training`, `evaluation`을 서로 독립된 plugin으로 만드는 것이다. `experiment.runner`는 YAML을 읽고 각 registry에서 필요한 객체를 조립하는 orchestration layer가 된다.

---

## 6. YAML configuration 설계

### 6.1 최상위 schema

권장 최상위 구조:

```yaml
schema_version: 1

experiment:
  name: wm811k_mobilenet_v3_colormap
  suite: wm811k_preprocess_ablation
  tags: [wm811k, classification, ablation]
  description: MobileNetV3-Small with colormap preprocessing.

fixed:
  dataset_id: wm811k_lswmd_v1
  split_id: wm811k_seed42_holdout_v1
  labels_id: wm811k_9class_paper_order
  metric:
    primary: macro_f1
    selection_split: val
  comparison_keys:
    - data.split
    - train.seed
    - train.max_epochs

data:
  module: wm811k
  source:
    path: ../LSWMD.pkl
    format: pandas_pickle
  labels:
    column: failureType
    aliases: paper
    include: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
  wafer:
    column: waferMap
  preprocessing:
    channel_mode: colormap
    image_size: 224
    normalize: none
  split:
    strategy: stratified_holdout
    seed: 42
    test_size: 0.2
    val_fraction_of_trainval: 0.25
    external_val: null
    external_test: null
    save_indices: true
  augmentation:
    enabled: true
    target_defect_count: 10000
    train_only: true
    transforms:
      random_crop:
        padding: 16
      random_rotation:
        degrees: 180
      horizontal_flip:
        p: 0.5
      vertical_flip:
        p: 0.5
      gaussian_blur:
        p: 0.2
        kernel_size: 3
      random_erasing:
        p: 0.25
        scale: [0.005, 0.04]
        ratio: [0.3, 3.3]
  dataloader:
    batch_size: 128
    num_workers: 0
    pin_memory: auto
    persistent_workers: false

task:
  type: classification
  num_classes: 9
  class_order: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
  loss:
    name: cross_entropy
    class_weights: none
  metrics:
    - accuracy
    - macro_precision
    - macro_recall
    - macro_f1
    - weighted_f1

model:
  name: mobilenet_v3_small
  source: torchvision
  pretrained: false
  dropout: 0.35
  num_classes: ${task.num_classes}

train:
  trainer: supervised_torch
  seed: 42
  max_epochs: 30
  optimizer:
    name: adam
    lr: 0.0001
    weight_decay: 0.0
  scheduler:
    name: none
  amp:
    enabled: false
  gradient_accumulation_steps: 1
  early_stopping:
    enabled: false
    monitor: val/macro_f1
    mode: max
    patience: 10
    min_delta: 0.0
  checkpoint:
    monitor: val/macro_f1
    mode: max
    save_best: true
    save_last: true

inference:
  batch_size: ${data.dataloader.batch_size}
  checkpoint: best
  output_probabilities: true
  top_k: 5
  threshold: null

evaluation:
  splits: [val, test]
  primary_metric: macro_f1
  save_predictions: true
  save_confusion_matrix: true
  save_curves: false

runtime:
  device: auto
  output_dir: outputs/experiments
  deterministic: true
  log_level: info
```

`${...}` 참조는 OmegaConf 스타일 예시이다. 꼭 OmegaConf를 써야 한다는 뜻은 아니지만, config inheritance와 interpolation을 고려하면 `omegaconf` 또는 `hydra-core` 도입이 유리하다. 가볍게 시작하려면 `pyyaml`과 dataclass validation을 조합해도 된다.

### 6.2 고정 파라미터와 변경 파라미터 분리

실험 비교에서 다음 항목은 자주 고정된다.

- raw dataset 경로와 dataset version
- label order와 label alias rule
- train/val/test split seed
- external validation/test set path
- test sample id 목록
- primary metric
- model selection split
- epoch 또는 training budget
- batch size
- optimizer family
- random seed 또는 seed list
- augmentation 적용 범위
- evaluation threshold 정책

반대로 다음 항목은 비교 축으로 바꾸기 좋다.

- `data.preprocessing.channel_mode`
- `data.preprocessing.image_size`
- `data.augmentation.target_defect_count`
- `model.name`
- `model.pretrained`
- `train.optimizer.lr`
- `train.scheduler.name`
- `train.early_stopping.enabled`
- `train.seed`
- `data.split.strategy`
- `task.loss.class_weights`

고정과 변경을 함께 관리하는 sweep 예시는 다음과 같다.

```yaml
schema_version: 1

sweep:
  name: wm811k_mxn_ablation
  base_config: configs/experiments/wm811k/001_paper_reproduction.yaml
  fixed:
    data.source.path: ../LSWMD.pkl
    data.split.seed: 42
    data.split.test_size: 0.2
    task.class_order: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
    train.max_epochs: 30
    train.optimizer.name: adam
    train.optimizer.lr: 0.0001
    evaluation.primary_metric: macro_f1

  axes:
    preprocessing:
      - name: colormap
        set:
          data.preprocessing.channel_mode: colormap
      - name: replicate
        set:
          data.preprocessing.channel_mode: replicate

    model:
      - name: mobilenet_v3_small
        set:
          model.name: mobilenet_v3_small
      - name: shufflenet_v2_x1_0
        set:
          model.name: shufflenet_v2_x1_0
      - name: resnet18
        set:
          model.name: resnet18

    training_control:
      - name: no_early_stop
        set:
          train.early_stopping.enabled: false
      - name: early_stop_patience10
        set:
          train.early_stopping.enabled: true
          train.early_stopping.patience: 10

  expansion:
    mode: grid
    include:
      - preprocessing
      - model
      - training_control

  repeats:
    seeds: [42, 43, 44]
    folds: null

  execution:
    max_parallel: 1
    continue_on_error: true
```

이 예시는 `2 x 3 x 2 x 3 seeds = 36` trial을 만든다. 여기서 `fixed` 값은 모든 trial에 공통으로 들어가며, 각 axis의 `set`만 바뀐다.

### 6.3 K-fold와 multi-seed 표현

K-fold와 seed 반복은 모델 성능의 분산을 보기 위해 매우 중요하다.

```yaml
data:
  split:
    strategy: stratified_kfold
    seed: 42
    n_splits: 4
    test_size: 0.2
    fold_index: ${trial.fold}

trial:
  seeds: [42, 43, 44, 45, 46]
  folds: [0, 1, 2, 3]
```

실행기는 이 설정을 받아 다음 trial들을 만든다.

```text
seed42_fold0
seed42_fold1
seed42_fold2
seed42_fold3
seed43_fold0
...
seed46_fold3
```

결과 집계는 평균과 표준편차를 최소 단위로 제공한다.

```text
model,preprocess,mean_macro_f1,std_macro_f1,n_trials
mobilenet_v3_small,colormap,0.895,0.006,20
mobilenet_v3_small,replicate,0.888,0.009,20
```

### 6.4 External validation/test set

사용자가 validation 또는 test set을 별도 경로로 고정하고 싶을 수 있다.

```yaml
data:
  split:
    strategy: external_test_with_train_val_split
    seed: 42
    external_test:
      path: data/splits/wm811k_test_ids.csv
      id_column: original_index
    external_val:
      path: null
    val_fraction_of_train: 0.2
```

규칙:

- external test IDs는 train/val 후보에서 반드시 제외한다.
- external val IDs가 있으면 train 후보에서 제외한다.
- ID가 raw dataset에 없으면 오류를 낸다.
- 같은 ID가 train/val/test에 중복되면 오류를 낸다.
- 생성된 split은 항상 `splits/*.csv`로 저장한다.

---

## 7. Experiment directory와 artifact 규약

권장 output 구조:

```text
outputs/
  experiments/
    wm811k_preprocess_ablation/
      20260518_142200_mobilenet_v3_colormap_a1b2c3d4/
        resolved_config.yaml
        source_config.yaml
        config_hash.txt
        run_manifest.json
        environment.json
        dataset_summary.json
        splits/
          train_base.csv
          train.csv
          val.csv
          test.csv
        preprocessing/
          manifest.json
        checkpoints/
          best.pt
          last.pt
        history.csv
        metrics/
          val_summary.json
          test_summary.json
          test_classification_report.csv
          test_classification_report.json
          test_confusion_matrix.csv
          test_confusion_matrix.png
          test_confusion_matrix_normalized.csv
          test_confusion_matrix_normalized.png
        predictions/
          test_predictions.csv
        logs/
          train.log
```

### 7.1 `run_manifest.json`

예시:

```json
{
  "experiment_name": "wm811k_mobilenet_v3_colormap",
  "suite": "wm811k_preprocess_ablation",
  "trial_name": "seed42_fold0",
  "config_hash": "a1b2c3d4",
  "started_at": "2026-05-18T14:22:00+09:00",
  "finished_at": "2026-05-18T15:10:13+09:00",
  "status": "completed",
  "primary_metric": "macro_f1",
  "best_epoch": 23,
  "checkpoint": "checkpoints/best.pt"
}
```

### 7.2 `environment.json`

포함하면 좋은 정보:

- Python version
- package versions
- OS
- device backend
- GPU name
- CUDA/cuDNN version
- CPU 정보
- seed
- deterministic 설정
- 현재 Git commit, Git diff dirty 여부
- raw dataset path
- raw dataset hash

현재 루트가 Git 저장소가 아닐 수 있으므로, Git 정보는 optional이어야 한다.

### 7.3 `predictions/test_predictions.csv`

classification 예시:

```text
sample_id,original_index,true_label,pred_label,prob_Center,prob_Donut,prob_Edge-loc,...
0,12345,None,None,0.001,0.000,...
```

time series forecasting 예시:

```text
sample_id,timestamp,target,prediction,horizon,split
series_001,2026-01-01T00:00:00,12.3,12.1,1,test
```

anomaly detection 예시:

```text
sample_id,true_label,anomaly_score,pred_label,threshold,split
abc,normal,0.12,normal,0.72,test
```

---

## 8. Core module 설계

### 8.1 `core.registry`

Registry는 문자열 이름을 구현체로 연결한다.

예상 역할:

- `data_module_registry`
- `model_registry`
- `task_registry`
- `trainer_registry`
- `evaluator_registry`
- `callback_registry`
- `optimizer_registry`
- `scheduler_registry`

개념 코드:

```python
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, key: str, value=None):
        def decorator(obj):
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._items[key] = obj
            return obj

        if value is not None:
            return decorator(value)
        return decorator

    def get(self, key: str):
        if key not in self._items:
            choices = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name}: {key}. Available: {choices}")
        return self._items[key]
```

이 구조가 있으면 새 모델을 추가할 때 `models.py`의 거대한 `if` 문을 계속 늘리지 않아도 된다.

### 8.2 `core.config`

책임:

- YAML load
- base config merge
- environment variable interpolation
- `${...}` reference resolution
- schema validation
- default materialization
- fixed override 검사
- config hash 생성
- `resolved_config.yaml` 저장

권장 함수:

```python
def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    ...

def resolve_config(config: RawConfig) -> ResolvedConfig:
    ...

def validate_config(config: ResolvedConfig) -> None:
    ...

def hash_config(config: ResolvedConfig, exclude_runtime: bool = True) -> str:
    ...
```

### 8.3 `core.random`

현재 `utils.set_seed`의 책임을 확장한다.

필수 기능:

- Python `random` seed
- NumPy seed
- PyTorch CPU/GPU seed
- deterministic cudnn 설정
- worker init seed
- trial seed와 data split seed 분리

중요한 구분:

- `data.split.seed`: split 재현성
- `train.seed`: model init, dataloader shuffle, augmentation randomness
- `sweep.seed`: random search sampling

세 seed를 분리해야 split을 고정한 채 학습 seed만 바꾸는 실험이 가능하다.

---

## 9. Data architecture

### 9.1 기본 interface

모든 데이터셋은 다음 역할을 구현한다.

```python
class BaseDataModule:
    def prepare_data(self) -> None:
        """Download, validate, or locate raw files."""

    def load_raw(self):
        """Load raw source into memory or an indexable representation."""

    def build_records(self):
        """Create canonical sample records with sample_id, label, source pointer."""

    def split(self):
        """Create train/val/test records."""

    def fit_preprocess(self, train_records):
        """Fit train-only preprocessing state."""

    def build_dataset(self, split: str):
        """Return Dataset object for train/val/test/infer."""

    def build_loader(self, split: str):
        """Return DataLoader or equivalent iterator."""

    def summary(self) -> dict:
        """Return dataset summary artifact."""
```

### 9.2 Canonical record schema

모든 modality의 기본 record는 다음 공통 컬럼을 가져야 한다.

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 플랫폼 내부의 고유 sample id |
| `source_id` | 원본 데이터 내부 id. WM-811K에서는 `original_index` |
| `label` | supervised task의 label. 비지도 task에서는 null 가능 |
| `split` | train, val, test, infer |
| `group_id` | grouped split이 필요한 경우 사용 |
| `weight` | sample weighting이 필요한 경우 사용 |
| `metadata` | JSON string 또는 별도 metadata path |

WM-811K record 추가 컬럼:

| 컬럼 | 의미 |
|---|---|
| `row_id` | filtered dataframe 내부 row index |
| `original_index` | LSWMD 원본 dataframe index |
| `failure_label` | 정규화된 defect label |
| `augmented` | augmentation으로 늘어난 record 여부 |
| `aug_seq` | augmentation sequence |

Time series record 추가 컬럼:

| 컬럼 | 의미 |
|---|---|
| `series_id` | 시계열 개체 id |
| `start_time` | window 시작 시각 |
| `end_time` | window 끝 시각 |
| `horizon` | forecasting horizon |

Image folder record 추가 컬럼:

| 컬럼 | 의미 |
|---|---|
| `image_path` | 이미지 파일 경로 |
| `width` | 원본 폭 |
| `height` | 원본 높이 |

### 9.3 Splitter

split strategy는 data module에서 분리한다.

지원해야 할 splitter:

- `stratified_holdout`
- `stratified_kfold`
- `group_kfold`
- `time_series_split`
- `external_split`
- `external_test_with_train_val_split`
- `predefined_column`
- `none`

Splitter interface:

```python
class BaseSplitter:
    def split(self, records, config) -> SplitBundle:
        ...
```

`SplitBundle`:

```python
@dataclass
class SplitBundle:
    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame | None
    metadata: dict
```

### 9.4 Preprocessing과 augmentation 분리

전처리와 augmentation은 다르다.

- preprocessing: deterministic transform 또는 train 통계에 fit되는 transform. 예: resize, normalization, missing value imputation, scaling.
- augmentation: 주로 train에서 stochastic하게 적용되는 변형. 예: random crop, random rotation, jitter.

WM-811K 현재 코드에서는 `build_transform` 안에서 resize와 augmentation이 함께 있다. 목표 구조에서는 다음처럼 나눈다.

```text
preprocess:
  wafer_to_rgb
  resize
  normalize

augment:
  random_crop
  random_rotation
  random_erasing
```

Leakage 방지 규칙:

- `fit_preprocess`는 train split만 본다.
- `transform_preprocess`는 train/val/test 모두에 적용한다.
- augmentation은 train dataset에서만 켠다.
- test time augmentation을 쓰고 싶으면 inference config에 별도 명시한다.

### 9.5 WM-811K DataModule 설계

현재 `data.py`를 다음으로 분리한다.

```text
datasets/wm811k/
  labels.py
    PAPER_CLASSES
    LABEL_ALIASES
    normalize_failure_label

  source.py
    load_lswmd
    find_column
    scalarize

  records.py
    build_labeled_records
    sample_per_class
    record_counts

  split.py
    make_single_split
    make_kfold_splits
    load_external_ids

  preprocess.py
    wafer_to_rgb_array
    build_preprocess_transform

  augment.py
    augment_training_records
    build_augmentation_transform

  dataset.py
    WaferMapDataset
    make_inference_tensor

  datamodule.py
    WM811KDataModule
```

이 분리는 다음 장점을 준다.

- label alias만 바꾸는 실험이 쉬워진다.
- split 정책만 바꾸는 실험이 쉬워진다.
- wafer map을 RGB로 바꾸는 방식만 비교하기 쉬워진다.
- 같은 training runner를 이미지 데이터셋에도 재사용할 수 있다.

### 9.6 다른 데이터 modality 확장

| Modality | DataModule 예 | Dataset 핵심 |
|---|---|---|
| Wafer map | `wm811k` | 2D bin map to tensor |
| 일반 이미지 | `image_folder` | image path, label, torchvision transform |
| 시계열 | `timeseries_window` | windowing, scaling, horizon |
| Tabular | `tabular_csv` | column schema, imputation, encoding |
| Sensor log | `sensor_sequence` | grouping, resampling, windowing |
| 강화학습 | `gym_env` | env factory, reset/step contract |

DataModule을 task와 분리하면 같은 시계열 데이터도 classification, forecasting, anomaly detection으로 다르게 사용할 수 있다.

---

## 10. Task architecture

### 10.1 왜 task 분리가 필요한가

현재 코드는 사실상 image classification 전용이다. 하지만 향후 목표에는 지도학습, 비지도학습, 강화학습이 모두 포함된다. 이들을 하나의 `train.py`에 조건문으로 넣으면 유지보수가 어렵다.

Task는 다음을 책임진다.

- label 또는 target 해석
- model output 해석
- loss 생성
- metric 생성
- prediction artifact schema
- best checkpoint selection metric

### 10.2 ClassificationTask

책임:

- `num_classes`
- `class_order`
- `CrossEntropyLoss`
- `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`
- confusion matrix
- classification report
- probability output

Config:

```yaml
task:
  type: classification
  num_classes: 9
  class_order: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
  loss:
    name: cross_entropy
    class_weights: balanced
  metrics:
    - accuracy
    - macro_f1
```

### 10.3 RegressionTask

Config:

```yaml
task:
  type: regression
  target_columns: [y]
  loss:
    name: mse
  metrics:
    - mae
    - rmse
    - r2
```

### 10.4 AnomalyDetectionTask

Config:

```yaml
task:
  type: anomaly_detection
  score_name: reconstruction_error
  threshold:
    strategy: val_f1_best
  metrics:
    - auroc
    - auprc
    - f1_at_threshold
```

### 10.5 ForecastingTask

Config:

```yaml
task:
  type: forecasting
  horizon: 24
  target_columns: [temperature]
  loss:
    name: mae
  metrics:
    - mae
    - rmse
    - smape
```

### 10.6 ReinforcementLearningTask

강화학습은 일반 supervised training loop와 다르므로 `Trainer`도 별도 plugin이어야 한다.

Config:

```yaml
task:
  type: reinforcement_learning
  objective: maximize_reward
  metrics:
    - mean_episode_reward
    - success_rate

train:
  trainer: ppo
  total_timesteps: 1000000
```

---

## 11. Model architecture

### 11.1 Model factory 목표

현재 `models.create_model`은 이름 기반 `if` 문으로 torchvision 모델을 만든다. 목표 구조에서는 registry 기반으로 바꾼다.

Model interface:

```python
class BaseModelFactory:
    def build(self, config, task) -> torch.nn.Module:
        ...
```

권장 registry:

```python
MODEL_REGISTRY = Registry("model")

@MODEL_REGISTRY.register("mobilenet_v3_small")
def build_mobilenet_v3_small(config, task):
    ...
```

### 11.2 모델 config 예시

Torchvision CNN:

```yaml
model:
  name: mobilenet_v3_small
  source: torchvision
  pretrained: false
  dropout: 0.35
  head:
    type: classification
    num_classes: ${task.num_classes}
```

Custom CNN:

```yaml
model:
  name: cnn_wdi
  source: local
  channels: [32, 64, 128, 192]
  dropout: 0.35
```

Time series Transformer:

```yaml
model:
  name: timeseries_transformer
  input_dim: 12
  d_model: 128
  num_layers: 4
  nhead: 8
  dropout: 0.1
```

AutoEncoder:

```yaml
model:
  name: conv_autoencoder
  latent_dim: 64
  reconstruction_loss: mse
```

RL policy:

```yaml
model:
  name: mlp_policy
  hidden_dims: [256, 256]
  activation: tanh
```

### 11.3 Checkpoint contract

Checkpoint에는 최소한 다음이 들어가야 한다.

```python
{
    "model_state": state_dict,
    "optimizer_state": optimizer_state,
    "scheduler_state": scheduler_state,
    "epoch": epoch,
    "global_step": global_step,
    "resolved_config": config_dict,
    "task_state": task_state,
    "labels": labels,
    "metric": best_metric,
}
```

모델을 load할 때 code default가 아니라 checkpoint 안의 `resolved_config`를 우선 사용한다.

---

## 12. Training architecture

### 12.1 Trainer 분리

현재 `train.py`는 모든 것을 수행한다. 목표는 다음 세 계층으로 나누는 것이다.

```text
CLI
  -> ExperimentRunner
      -> DataModule
      -> Task
      -> ModelFactory
      -> Trainer
          -> callbacks
          -> evaluator
```

`Trainer`는 학습 loop만 책임진다.

Supervised trainer interface:

```python
class SupervisedTorchTrainer:
    def fit(self, model, datamodule, task, config, callbacks) -> TrainResult:
        ...

    def train_epoch(self, model, loader) -> dict:
        ...

    def validate(self, model, loader) -> dict:
        ...
```

### 12.2 Callback 체계

지원할 callback:

- `CheckpointCallback`
- `EarlyStoppingCallback`
- `HistoryLogger`
- `LRSchedulerCallback`
- `GradientClippingCallback`
- `PredictionSampleCallback`
- `ConfusionMatrixCallback`
- `TimeLimitCallback`

Early stopping config:

```yaml
train:
  early_stopping:
    enabled: true
    monitor: val/macro_f1
    mode: max
    patience: 10
    min_delta: 0.0001
    restore_best: true
```

Checkpoint config:

```yaml
train:
  checkpoint:
    monitor: val/macro_f1
    mode: max
    save_best: true
    save_last: true
    save_every_n_epochs: null
```

### 12.3 Optimizer와 scheduler

Optimizer registry:

```yaml
train:
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01
```

Scheduler 예:

```yaml
train:
  scheduler:
    name: cosine_annealing
    t_max: 30
    eta_min: 0.000001
```

또는:

```yaml
train:
  scheduler:
    name: reduce_on_plateau
    monitor: val/macro_f1
    mode: max
    factor: 0.5
    patience: 5
```

### 12.4 Multi-seed

Seed 반복은 sweep runner가 trial을 확장하는 방식으로 처리한다.

```yaml
trial:
  seeds: [42, 43, 44, 45, 46]
```

각 trial은 다음을 별도로 저장한다.

- `train.seed`
- `data.split.seed`
- `trial.seed`
- `run_manifest.trial_name`

Split seed를 고정하고 train seed만 바꾸려면:

```yaml
data:
  split:
    seed: 42

trial:
  seeds: [1, 2, 3, 4, 5]
```

### 12.5 K-fold

K-fold는 data split strategy와 trial fold index의 조합으로 처리한다.

```yaml
data:
  split:
    strategy: stratified_kfold
    n_splits: 4
    test_size: 0.2
    seed: 42

trial:
  folds: [0, 1, 2, 3]
```

각 fold의 test set이 동일해야 하는 paper-style split에서는 holdout test를 먼저 만들고 trainval에 대해 K-fold를 적용한다. 현재 `make_kfold_splits`가 이 로직을 이미 구현하고 있으므로 이를 splitter plugin으로 옮기면 된다.

### 12.6 Training budget 공정성

실험 비교에서는 다음 중 하나를 고정해야 한다.

- epoch 수
- optimizer step 수
- wall-clock time
- processed sample 수

WM-811K augmentation을 바꾸면 train record 수가 달라질 수 있으므로 epoch 기준 비교가 불공정할 수 있다. 문서화가 필요하다.

권장:

- 논문 재현은 `max_epochs` 고정
- augmentation 개수 비교는 `max_steps` 또는 `processed_samples`도 함께 기록
- early stopping 사용 시 `best_epoch`, `trained_epochs`, `total_steps`를 비교표에 포함

---

## 13. Inference architecture

추론은 학습과 분리되어야 한다. 같은 checkpoint로 다음 입력을 처리할 수 있어야 한다.

- stored split 전체
- 별도 CSV/폴더에 있는 batch input
- 단일 sample
- API 또는 notebook에서 들어오는 in-memory input

Inference config:

```yaml
inference:
  checkpoint: outputs/experiments/.../checkpoints/best.pt
  input:
    type: split
    split: test
  output:
    dir: outputs/inference/my_run
    save_probabilities: true
    save_visualizations: true
  postprocess:
    top_k: 5
    threshold: null
```

WM-811K 단일 wafer 추론:

```yaml
inference:
  input:
    type: wm811k_row
    data_path: ../LSWMD.pkl
    original_index: 12345
```

시계열 batch 추론:

```yaml
inference:
  input:
    type: timeseries_csv
    path: data/new_sensor_windows.csv
```

---

## 14. Evaluation architecture

### 14.1 Classification metrics

필수 metric:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- per-class precision/recall/F1
- confusion matrix
- normalized confusion matrix

선택 metric:

- top-k accuracy
- ROC-AUC
- PR-AUC
- calibration error
- balanced accuracy
- Cohen's kappa

WM-811K처럼 class imbalance가 큰 경우에는 macro F1을 primary metric으로 두는 것이 적절하다.

### 14.2 Aggregation

Fold/seed를 포함한 결과 집계 schema:

```text
suite,experiment,axis_preprocess,axis_model,seed,fold,accuracy,macro_f1,best_epoch
wm811k_ablation,e001,colormap,mobilenet_v3_small,42,0,0.980,0.895,23
```

집계 파일:

- `comparison_trials.csv`: trial별 raw metric
- `comparison_grouped.csv`: axis 조합별 mean/std/min/max
- `leaderboard.csv`: primary metric 기준 정렬

Grouped metric:

```text
axis_preprocess,axis_model,n,macro_f1_mean,macro_f1_std,macro_f1_min,macro_f1_max
colormap,mobilenet_v3_small,20,0.895,0.006,0.884,0.904
```

### 14.3 공정 비교 규칙

비교표 생성기는 다음을 검사해야 한다.

- 같은 raw dataset인가
- 같은 test set인가
- 같은 split seed인가
- 같은 label order인가
- 같은 primary metric인가
- 같은 training budget인가
- 같은 external test path인가
- 같은 fold/seed 조합인가

다르면 비교표에 warning을 추가한다.

---

## 15. Sweep 설계

### 15.1 Grid sweep

모든 축의 Cartesian product를 실행한다.

```yaml
sweep:
  expansion:
    mode: grid
```

장점:

- 모든 조합을 빠짐없이 비교
- ablation과 논문 표 작성에 좋음

단점:

- 축이 많으면 실행 수가 폭발

### 15.2 Manual sweep

연구자가 직접 조합을 적는다.

```yaml
sweep:
  expansion:
    mode: manual
  trials:
    - name: baseline
      set:
        model.name: mobilenet_v3_small
        data.preprocessing.channel_mode: colormap
    - name: no_aug
      set:
        model.name: mobilenet_v3_small
        data.preprocessing.channel_mode: colormap
        data.augmentation.enabled: false
```

장점:

- 비교하고 싶은 조합만 실행
- 실험 비용 관리가 쉬움

### 15.3 Random sweep

연속형 hyperparameter를 일부 sampling한다.

```yaml
sweep:
  expansion:
    mode: random
    n_trials: 30
    seed: 2026
  search_space:
    train.optimizer.lr:
      type: loguniform
      low: 0.00001
      high: 0.001
    train.weight_decay:
      type: choice
      values: [0.0, 0.0001, 0.001]
```

초기 구현에서는 grid/manual만으로 충분하고, random/Optuna는 후순위로 둘 수 있다.

---

## 16. CLI 설계

권장 명령:

```powershell
python -m wafer_repro.cli.validate_config --config configs/experiments/wm811k/001_paper_reproduction.yaml
python -m wafer_repro.cli.run --config configs/experiments/wm811k/001_paper_reproduction.yaml
python -m wafer_repro.cli.sweep --config configs/sweeps/wm811k_mxn_grid.yaml
python -m wafer_repro.cli.evaluate --run-dir outputs/experiments/... --split test
python -m wafer_repro.cli.infer --config configs/inference/wm811k_single.yaml
python -m wafer_repro.cli.compare --suite-dir outputs/experiments/wm811k_preprocess_ablation
```

CLI override:

```powershell
python -m wafer_repro.cli.run `
  --config configs/experiments/wm811k/001_paper_reproduction.yaml `
  --set train.max_epochs=5 `
  --set runtime.device=cpu
```

주의:

- override는 debugging에는 편하지만 최종 실험에서는 resolved config에 반드시 저장되어야 한다.
- fixed key override는 기본적으로 막는다.

---

## 17. 현재 코드 기준 마이그레이션 계획

### Phase 0. 문서화와 현 코드 보존

목표:

- 현재 논문 재현 코드가 계속 실행되도록 둔다.
- 새 사양서를 기준으로 단계별 리팩터링을 시작한다.

작업:

- 이 문서 추가
- 기존 README에 이 문서 링크 추가
- 현재 CLI 실행 예시는 유지

### Phase 1. YAML config loader 도입

목표:

- 기존 CLI argument와 동일한 값을 YAML로 받을 수 있게 한다.
- 내부 학습 로직은 최대한 그대로 둔다.

작업:

- `core/config.py` 추가
- `configs/experiments/wm811k/001_paper_reproduction.yaml` 추가
- `train.py`에 `--config` option 추가
- CLI argument와 config merge 규칙 정의
- `resolved_config.yaml` 저장

이 단계에서는 구조가 완전히 깔끔하지 않아도 된다. 가장 중요한 것은 실험 조건이 파일로 남는 것이다.

### Phase 2. data module 분리

목표:

- `data.py`의 책임을 WM-811K DataModule로 분리한다.

작업:

- `datasets/wm811k/source.py`
- `datasets/wm811k/labels.py`
- `datasets/wm811k/split.py`
- `datasets/wm811k/dataset.py`
- `datasets/wm811k/datamodule.py`
- 기존 함수는 compatibility wrapper로 유지

완료 기준:

- 기존 `python -m wafer_repro.train ...`가 그대로 동작
- 새 `DataModule` 경유 실행도 동작

### Phase 3. model/task/trainer registry 도입

목표:

- 모델과 task와 trainer를 config 이름으로 조립한다.

작업:

- `core/registry.py`
- `tasks/classification.py`
- `models/registry.py`
- `training/supervised.py`
- 기존 `models.create_model`은 registry wrapper로 전환

### Phase 4. ExperimentRunner 도입

목표:

- `train.py`의 orchestration을 `experiment/runner.py`로 옮긴다.

작업:

- `ExperimentRunner.run(config)`
- run directory 생성
- artifact 저장 표준화
- environment capture
- train/evaluate/infer 흐름 분리

### Phase 5. Sweep과 comparison

목표:

- `run_experiments.py`를 범용 sweep runner로 대체한다.

작업:

- `experiment/sweep.py`
- grid/manual sweep expansion
- fixed override 검사
- trial manifest 저장
- `analysis/collector.py`
- `analysis/leaderboard.py`

### Phase 6. 새로운 modality 추가

목표:

- WM-811K 외 데이터로 구조 검증

권장 순서:

1. `image_folder` classification
2. `tabular_csv` classification/regression
3. `timeseries_window` forecasting/classification
4. anomaly detection
5. reinforcement learning

이 순서가 좋은 이유는 supervised classification runner를 먼저 안정화한 뒤, 점점 다른 task contract를 추가할 수 있기 때문이다.

---

## 18. WM-811K 실험 축 예시

### 18.1 전처리 축

| Axis | Method | YAML key |
|---|---|---|
| channel mapping | colormap | `data.preprocessing.channel_mode: colormap` |
| channel mapping | replicate | `data.preprocessing.channel_mode: replicate` |
| image size | 64 | `data.preprocessing.image_size: 64` |
| image size | 128 | `data.preprocessing.image_size: 128` |
| image size | 224 | `data.preprocessing.image_size: 224` |
| normalization | none | `data.preprocessing.normalize: none` |
| normalization | imagenet | `data.preprocessing.normalize: imagenet` |

### 18.2 Split 축

| Axis | Method | YAML key |
|---|---|---|
| split | single 6:2:2 | `data.split.strategy: stratified_holdout` |
| split | paper 4-fold | `data.split.strategy: stratified_kfold` |
| split | external test | `data.split.strategy: external_test_with_train_val_split` |
| seed | 42 | `data.split.seed: 42` |
| seed | multi | `trial.seeds: [42, 43, 44]` |

### 18.3 Augmentation 축

| Axis | Method | YAML key |
|---|---|---|
| augmentation | off | `data.augmentation.enabled: false` |
| target count | 5000 | `data.augmentation.target_defect_count: 5000` |
| target count | 10000 | `data.augmentation.target_defect_count: 10000` |
| rotation | 90 | `data.augmentation.transforms.random_rotation.degrees: 90` |
| rotation | 180 | `data.augmentation.transforms.random_rotation.degrees: 180` |
| erasing | off | `data.augmentation.transforms.random_erasing.p: 0.0` |
| erasing | paper-like | `data.augmentation.transforms.random_erasing.p: 0.25` |

### 18.4 Model 축

| Axis | Method |
|---|---|
| model | `resnet18` |
| model | `efficientnet_v2_s` |
| model | `shufflenet_v2_x1_0` |
| model | `shufflenet_v2_x0_5` |
| model | `mobilenet_v2` |
| model | `mobilenet_v3_small` |
| model | `cnn_wdi` |
| model | future custom wafer transformer |

### 18.5 Training control 축

| Axis | Method | YAML key |
|---|---|---|
| optimizer | Adam | `train.optimizer.name: adam` |
| optimizer | AdamW | `train.optimizer.name: adamw` |
| lr | 1e-4 | `train.optimizer.lr: 0.0001` |
| scheduler | none | `train.scheduler.name: none` |
| scheduler | cosine | `train.scheduler.name: cosine_annealing` |
| early stopping | off | `train.early_stopping.enabled: false` |
| early stopping | patience 10 | `train.early_stopping.patience: 10` |
| class weights | none | `task.loss.class_weights: none` |
| class weights | balanced | `task.loss.class_weights: balanced` |

---

## 19. 예시 YAML: 논문 재현 baseline

```yaml
schema_version: 1

experiment:
  name: wm811k_paper_mobilenet_v3_small
  suite: wm811k_paper_reproduction
  tags: [wm811k, paper, mobilenetv3]

fixed:
  dataset_id: wm811k_lswmd
  labels_id: paper_9_classes
  metric:
    primary: macro_f1
    selection_split: val

data:
  module: wm811k
  source:
    path: ../LSWMD.pkl
  preprocessing:
    channel_mode: colormap
    image_size: 224
  split:
    strategy: stratified_kfold
    seed: 42
    test_size: 0.2
    n_splits: 4
    fold_index: 0
  augmentation:
    enabled: true
    target_defect_count: 10000
    train_only: true
    transforms:
      random_crop:
        padding: 16
      random_rotation:
        degrees: 180
      gaussian_blur:
        p: 0.2
      random_erasing:
        p: 0.25
  dataloader:
    batch_size: 128
    num_workers: 0

task:
  type: classification
  class_order: [Center, Donut, Edge-loc, Edge-ring, Loc, Near-full, Random, Scratch, None]
  loss:
    name: cross_entropy
    class_weights: none
  metrics: [accuracy, macro_precision, macro_recall, macro_f1, weighted_f1]

model:
  name: mobilenet_v3_small
  pretrained: false
  dropout: 0.35

train:
  trainer: supervised_torch
  seed: 42
  max_epochs: 30
  optimizer:
    name: adam
    lr: 0.0001
    weight_decay: 0.0
  scheduler:
    name: none
  amp:
    enabled: false
  early_stopping:
    enabled: false
  checkpoint:
    monitor: val/macro_f1
    mode: max

evaluation:
  splits: [test]
  primary_metric: macro_f1
  save_predictions: true
  save_confusion_matrix: true

runtime:
  device: auto
  output_dir: outputs/experiments
  deterministic: true
```

---

## 20. 예시 YAML: 전처리 x 모델 x early stopping grid

```yaml
schema_version: 1

sweep:
  name: wm811k_preprocess_model_earlystop_grid
  base_config: configs/experiments/wm811k/001_paper_reproduction.yaml

  fixed:
    data.source.path: ../LSWMD.pkl
    data.split.strategy: stratified_holdout
    data.split.seed: 42
    data.split.test_size: 0.2
    data.split.val_fraction_of_trainval: 0.25
    train.max_epochs: 30
    task.loss.name: cross_entropy
    evaluation.primary_metric: macro_f1

  axes:
    preprocess:
      - name: colormap
        set:
          data.preprocessing.channel_mode: colormap
      - name: replicate
        set:
          data.preprocessing.channel_mode: replicate

    model:
      - name: mobilenet_v3_small
        set:
          model.name: mobilenet_v3_small
      - name: mobilenet_v2
        set:
          model.name: mobilenet_v2
      - name: shufflenet_v2_x1_0
        set:
          model.name: shufflenet_v2_x1_0

    early_stopping:
      - name: off
        set:
          train.early_stopping.enabled: false
      - name: patience10
        set:
          train.early_stopping.enabled: true
          train.early_stopping.monitor: val/macro_f1
          train.early_stopping.mode: max
          train.early_stopping.patience: 10

  expansion:
    mode: grid

  repeats:
    seeds: [42, 43, 44]

  execution:
    max_parallel: 1
    continue_on_error: true
```

---

## 21. 코드별 상세 리팩터링 설명

### 21.1 `data.py`

현재 `data.py`는 매우 많은 책임을 갖고 있다.

현재 함수별 의미:

- `_find_column`: 후보 컬럼명 중 실제 dataframe에 있는 컬럼을 찾는다.
- `scalarize`: `np.array([['Center']])` 같은 중첩 label 값을 문자열 하나로 편다.
- `normalize_failure_label`: label alias를 paper class 이름으로 정규화한다.
- `load_lswmd`: pickle 로드, wafer/failure column 탐색, label 정규화, labeled row filtering을 수행한다.
- `sample_per_class`: smoke test용 class별 샘플 수 제한을 한다.
- `base_records`: dataframe row를 학습용 record table로 바꾼다.
- `make_single_split`: stratified train/val/test split을 만든다.
- `make_kfold_splits`: holdout test를 만든 뒤 trainval에 대해 stratified k-fold를 만든다.
- `augment_training_records`: defect class만 target count까지 oversampling record를 만든다.
- `wafer_to_rgb_array`: wafer map 0/1/2 값을 RGB image로 변환한다.
- `build_transform`: torchvision transform을 만든다.
- `WaferMapDataset`: record를 tensor와 target으로 바꾼다.
- `make_inference_tensor`: 단일 wafer map을 inference tensor로 만든다.

리팩터링 방향:

- raw load와 label normalize는 `datasets/wm811k/source.py`, `labels.py`로 이동한다.
- split 함수는 공통 splitter로 이동하되 WM-811K paper split은 dataset plugin에서 config preset으로 제공한다.
- augmentation record 생성은 oversampling policy로 분리한다.
- image transform은 preprocessing과 augmentation으로 분리한다.
- `WaferMapDataset`은 가능한 얇게 유지한다.

### 21.2 `train.py`

현재 `train.py`는 다음 전체 흐름을 수행한다.

1. CLI argument parsing
2. seed 설정
3. device 선택
4. run directory 생성
5. LSWMD 로드
6. split 생성
7. augmentation record 생성
8. split CSV 저장
9. dataset과 dataloader 생성
10. data summary 저장
11. config 저장
12. model 생성
13. loss와 optimizer 생성
14. epoch loop 실행
15. best/last checkpoint 저장
16. best checkpoint test 평가

목표 구조에서는 `train.py`가 얇은 CLI wrapper가 된다.

```python
def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.set)
    runner = ExperimentRunner(config)
    runner.run()
```

실제 로직은 다음으로 이동한다.

- run directory: `experiment/manifest.py`
- data 준비: `DataModule`
- model 생성: `ModelRegistry`
- loss/metric: `Task`
- 학습 loop: `Trainer`
- checkpoint: `CheckpointCallback`
- 평가: `Evaluator`

### 21.3 `models.py`

현재 장점:

- paper model 목록이 명확하다.
- torchvision 모델의 classifier 교체가 잘 캡슐화되어 있다.
- `CNNWDIStyle`과 `SmallCNN`이 local baseline으로 존재한다.

한계:

- 새 모델을 추가할수록 `create_model`의 `if` 문이 길어진다.
- task가 classification이라는 가정이 강하다.
- input channel, output head, pretrained weight policy가 제한적이다.

리팩터링 방향:

- `MODEL_REGISTRY` 도입
- `build_torchvision_classifier` helper 도입
- model config를 그대로 checkpoint에 저장
- task가 `num_classes` 또는 output dimension을 제공

### 21.4 `metrics.py`

현재 장점:

- classification report와 confusion matrix를 잘 저장한다.
- normalized confusion matrix까지 생성한다.

한계:

- classification 전용이다.
- prediction CSV가 표준 artifact로 저장되지 않는다.
- metric name namespace가 없다. 예: `val/macro_f1`, `test/macro_f1`.

리팩터링 방향:

- `evaluation/classification.py`로 이동
- `ClassificationEvaluator.evaluate(y_true, y_prob)` 형태로 정리
- `summary.json`뿐 아니라 `predictions.csv`도 저장
- split prefix를 metric key에 반영

### 21.5 `run_experiments.py`

현재는 paper model x fold를 subprocess로 실행한다. 이 방식은 단순하고 안정적이지만 확장성이 제한된다.

목표:

- sweep YAML을 읽어 trial list 생성
- 각 trial별 resolved config 생성
- fixed override 검사
- 병렬 실행 옵션
- 실패 trial 기록
- 재시작 기능

Trial manifest 예:

```json
{
  "trial_name": "preprocess=colormap__model=mobilenet_v3_small__seed=42",
  "status": "pending",
  "config_path": "outputs/.../resolved_config.yaml"
}
```

---

## 22. 고정 파라미터 기능 설계

### 22.1 Fixed namespace

`fixed`는 비교의 기준을 명시한다. 실제 실행 config와 중복되는 값도 일부러 적는다. 이는 사람과 코드 모두에게 비교 기준을 드러내기 위함이다.

예:

```yaml
fixed:
  data.split.seed: 42
  data.split.test_size: 0.2
  train.max_epochs: 30
  evaluation.primary_metric: macro_f1
```

검증 규칙:

- `fixed`에 있는 key가 resolved config에 없으면 오류
- `fixed` 값과 resolved config 값이 다르면 오류
- sweep axis가 fixed key를 바꾸려 하면 오류

### 22.2 Frozen split artifact

split을 완전히 고정하려면 seed보다 split file을 저장하고 재사용하는 것이 더 강하다.

```yaml
data:
  split:
    strategy: predefined_files
    files:
      train: data/splits/wm811k_seed42/train.csv
      val: data/splits/wm811k_seed42/val.csv
      test: data/splits/wm811k_seed42/test.csv
```

이 방식은 다음 경우에 좋다.

- 여러 코드 버전에서 같은 split을 강제하고 싶을 때
- seed 기반 split이 library version 차이에 영향을 받을 수 있을 때
- test set을 절대 바꾸면 안 되는 benchmark를 만들 때

### 22.3 Dataset hash

dataset path만 저장하면 파일이 바뀌었는지 알 수 없다. 가능하면 raw file hash를 저장한다.

```json
{
  "path": "../LSWMD.pkl",
  "sha256": "...",
  "size_bytes": 123456789
}
```

큰 파일 hash 계산이 부담되면 optional로 두고, 최소한 size와 modified time은 저장한다.

---

## 23. 확장 예시: 시계열 데이터

시계열 데이터를 추가할 때 새로 필요한 것은 `DataModule`, model, task config뿐이다. training runner는 가능하면 재사용한다.

Config 예:

```yaml
data:
  module: timeseries_window
  source:
    path: data/sensors.csv
    time_column: timestamp
    group_column: machine_id
  preprocessing:
    resample: 1min
    scaler: standard
    fit_on: train
  window:
    length: 128
    stride: 16
    horizon: 1
  split:
    strategy: time_series_split
    val_size: 0.1
    test_size: 0.2

task:
  type: forecasting
  target_columns: [sensor_y]
  metrics: [mae, rmse]

model:
  name: timeseries_transformer
  input_dim: 12
  d_model: 128

train:
  trainer: supervised_torch
  max_epochs: 50
```

중요한 점:

- time series split은 미래 데이터가 train에 들어가지 않도록 시간 순서를 지켜야 한다.
- scaler는 train 구간에만 fit해야 한다.
- group별 leakage를 막으려면 `group_column`을 split에 반영해야 한다.

---

## 24. 확장 예시: 이미지 폴더 데이터

```yaml
data:
  module: image_folder
  source:
    root: data/images
    layout: class_folders
  preprocessing:
    image_size: 224
    normalize: imagenet
  split:
    strategy: stratified_holdout
    seed: 42
    test_size: 0.2
    val_fraction_of_trainval: 0.25
  augmentation:
    enabled: true
    transforms:
      random_resized_crop:
        scale: [0.8, 1.0]
      horizontal_flip:
        p: 0.5

task:
  type: classification

model:
  name: efficientnet_v2_s
  pretrained: true
```

이 경우 `WM811KDataModule`이 아니라 `ImageFolderDataModule`만 새로 구현하면 된다.

---

## 25. 확장 예시: 비지도 이상탐지

```yaml
data:
  module: image_folder
  source:
    root: data/normal_only_train
  split:
    strategy: external_test_with_train_val_split
    external_test:
      path: data/anomaly_test.csv

task:
  type: anomaly_detection
  label_semantics:
    normal: 0
    anomaly: 1
  threshold:
    strategy: maximize_val_f1

model:
  name: conv_autoencoder
  latent_dim: 64

train:
  trainer: reconstruction
  max_epochs: 100
```

이때 supervised classification과 달리 model output은 class logits가 아니라 reconstruction이다. 따라서 task와 evaluator가 output을 anomaly score로 변환한다.

---

## 26. 실험 비교 리포트 설계

비교 리포트는 단순 CSV 외에 Markdown 또는 HTML로도 생성할 수 있다.

Markdown 리포트 구성:

```text
# Suite Report: wm811k_preprocess_ablation

## Summary
- Dataset: wm811k_lswmd_v1
- Fixed split: seed 42, test_size 0.2
- Primary metric: test/macro_f1
- Trials: 36 completed, 0 failed

## Leaderboard
| rank | preprocess | model | early_stop | macro_f1_mean | macro_f1_std |

## Per-axis Analysis
### preprocessing
### model
### early_stopping

## Warnings
- All trials share the same test split.
- Seeds: 42, 43, 44.
```

축별 영향 분석:

- 같은 model, seed, split에서 preprocess만 바꾼 paired difference
- 같은 preprocess, seed, split에서 model만 바꾼 paired difference
- early stopping on/off의 best epoch와 test metric 비교

가능한 산출:

```text
axis,method_a,method_b,mean_delta_macro_f1,std_delta,n_pairs
preprocess,colormap,replicate,0.007,0.003,9
early_stop,off,patience10,0.002,0.004,9
```

이런 paired 비교는 단순 mean 비교보다 어떤 방향의 변화가 실제로 도움이 되는지 더 잘 보여준다.

---

## 27. Validation과 오류 처리

Config validation에서 잡아야 할 것:

- `data.module`이 registry에 없음
- `model.name`이 registry에 없음
- `task.type`이 trainer와 호환되지 않음
- classification `num_classes`와 `class_order` 길이가 다름
- external split file path가 없음
- split에 sample 중복 존재
- fixed key와 config 값이 다름
- primary metric이 evaluator output에 없음
- early stopping monitor metric이 validation에서 계산되지 않음
- checkpoint monitor mode가 `min`/`max` 중 하나가 아님
- device가 사용 불가능함

오류 메시지는 실행 중간이 아니라 시작 전에 최대한 알려야 한다.

---

## 28. 추가 의존성 제안

최소 도입:

- `pyyaml`: YAML parsing

권장 도입:

- `omegaconf`: config merge, interpolation
- `pydantic`: schema validation
- `rich`: CLI log와 table 표시
- `pyarrow`: prediction/result parquet 저장

나중에 고려:

- `hydra-core`: config composition과 sweep 관리
- `optuna`: hyperparameter optimization
- `mlflow` 또는 `wandb`: 외부 experiment tracking
- `lightning`: trainer 추상화. 다만 처음부터 도입하면 기존 코드와 차이가 커질 수 있으므로 신중히 결정한다.

초기 리팩터링에서는 `pyyaml + dataclass`만으로 시작해도 충분하다.

---

## 29. 구현 우선순위 체크리스트

가장 먼저 구현하면 좋은 순서:

1. `docs/experiment_platform_spec.md` 작성
2. `configs/experiments/wm811k/001_paper_reproduction.yaml` 추가
3. `core/config.py` 추가
4. `train.py --config` 지원
5. `resolved_config.yaml` 저장
6. `environment.json` 저장
7. split file 재사용 기능 추가
8. early stopping callback 추가
9. `experiment/sweep.py`로 grid sweep 구현
10. `analysis/collector.py`로 suite 결과 집계
11. `data.py`를 WM-811K DataModule로 분리
12. model registry 도입
13. task/evaluator registry 도입
14. image folder DataModule 추가
15. time series DataModule 추가

이 순서의 장점은 실험 재현성과 비교 기능을 먼저 얻고, 그 다음 내부 구조를 차근차근 넓힐 수 있다는 점이다.

---

## 30. 최종 목표 그림

최종적으로 사용자는 다음 흐름으로 실험할 수 있어야 한다.

```powershell
python -m wafer_repro.cli.sweep --config configs/sweeps/wm811k_mxn_grid.yaml
python -m wafer_repro.cli.compare --suite-dir outputs/experiments/wm811k_mxn_grid
```

그리고 새 데이터셋을 추가할 때는 다음만 작성하면 된다.

```text
datasets/my_dataset/datamodule.py
configs/datasets/my_dataset.yaml
configs/experiments/my_dataset/baseline.yaml
```

새 모델을 추가할 때는 다음만 작성하면 된다.

```text
models/my_model.py
@MODEL_REGISTRY.register("my_model")
```

새 평가 지표를 추가할 때는 다음만 작성하면 된다.

```text
evaluation/my_metric.py
@METRIC_REGISTRY.register("my_metric")
```

이 구조가 완성되면 프로젝트는 WM-811K 논문 재현 코드에서 출발했지만, 이후에는 다음 질문을 실험으로 답할 수 있는 플랫폼이 된다.

- 같은 데이터에서 어떤 전처리가 가장 중요한가?
- 모델 구조 변경이 실제로 도움이 되는가?
- 성능 개선이 특정 split이나 seed에만 의존하는가?
- early stopping, scheduler, class weighting이 class imbalance 문제를 완화하는가?
- 새로운 데이터 modality에서도 같은 실험 runner를 재사용할 수 있는가?
- 지도학습 외 task에서도 artifact와 비교 체계를 유지할 수 있는가?

이 문서는 그 전환을 위한 기준 사양서이며, 이후 실제 리팩터링은 이 문서의 phase 순서대로 진행하는 것을 권장한다.
