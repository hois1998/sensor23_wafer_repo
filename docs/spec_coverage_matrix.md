# Spec Coverage Matrix

이 문서는 `docs/experiment_platform_spec.md` 1-30장 전체를 기준으로 현재 구현 상태를 추적한다.

기존 Phase 0-10은 주로 17장의 마이그레이션 계획과 그 직후 hardening을 구현한 것이다. 앞으로의 Phase는 17장만이 아니라 사양서 전체 장을 기준으로 정의한다.

## 상태 정의

| 상태 | 의미 |
|---|---|
| 완료 | 사양서 요구사항이 현재 코드, 설정, artifact, 검증 로그에 실질적으로 반영됨 |
| 부분 | 핵심 골격은 있으나 사양서의 일부 요구사항, 범용성, 검증, artifact 규약이 남아 있음 |
| 미구현 | 아직 코드나 설정으로 구현되지 않았고 향후 Phase 대상임 |
| 문서 | 주로 방향성/용어/예시를 제공하며 별도 코드 구현보다는 다른 Phase의 기준 역할을 함 |

## 1-30장 커버리지 요약

| 장 | 주제 | 현재 상태 | 현재 반영 위치 | 남은 작업 | 후속 Phase |
|---:|---|---|---|---|---|
| 1 | 현재 프로젝트의 성격 | 완료 | `docs/experiment_platform_spec.md`, `docs/implementation_log.md` | 없음 | - |
| 2 | 설계 원칙 | 부분 | `core/config.py`, `experiment/runner.py`, `experiment/sweep.py`, `datasets/registry.py` | artifact contract, leakage 검증, fixed/frozen control 강화 | 12, 13 |
| 3 | 용어 정의 | 문서 | `docs/experiment_platform_spec.md` | 실제 코드 타입명과 문서 용어 일부 정렬 | 22 |
| 4 | 현재 코드와 목표 구조 매핑 | 부분 | `data.py` facade, `datasets/*`, `experiment/*`, `training/*`, `analysis/*` | `evaluation/`, `inference/`, `benchmark/`, `cli/` 패키지 정리 | 14, 22 |
| 5 | 권장 패키지 구조 | 부분 | 현재 `core/`, `experiment/`, `datasets/`, `tasks/`, `training/`, `analysis/` 존재 | `evaluation/`, `inference/`, `benchmark/`, `data/` 공통 계층, `cli/` 계층 미정리 | 14, 22 |
| 6 | YAML configuration 설계 | 부분 | `core/config.py`, `validate_config.py`, `configs/experiments/*`, `configs/sweeps/*` | schema validation, config composition, fixed/axes 충돌 검사 강화 | 12 |
| 7 | Experiment directory와 artifact 규약 | 부분 | `resolved_config.yaml`, `config_hash.txt`, `environment.json`, `run_manifest.json`, `history.csv`, split CSV, metrics JSON | prediction artifact, dataset hash, preprocessing manifest, checkpoint directory 규약 | 13 |
| 8 | Core module 설계 | 부분 | `core/registry.py`, `core/config.py`, `core/environment.py`, `utils.py` | random/device/io/hashing을 core로 더 분리 | 22 |
| 9 | Data architecture | 부분 | `datasets/base.py`, `datasets/registry.py`, `datasets/wm811k/*`, `datasets/image_folder/*` | canonical record 검증, preprocessing/augmentation 분리 강화, tabular/timeseries 추가 | 13, 18, 19 |
| 10 | Task architecture | 부분 | `tasks/classification.py`, `tasks/registry.py` | regression, anomaly detection, forecasting, RL task 미구현 | 20, 21 |
| 11 | Model architecture | 부분 | `models.py`, `MODEL_REGISTRY`, checkpoint 내 `model_state`, labels/config 저장 | 모델 패키지 분리, checkpoint contract 명문화, task별 output contract | 13, 22 |
| 12 | Training architecture | 부분 | `training/supervised.py`, `training/callbacks.py`, optimizer/scheduler/early stopping | gradient accumulation, resume training, multi-seed, k-fold suite, budget fairness | 15, 16 |
| 13 | Inference architecture | 부분 | `infer.py` WM-811K/image-folder 지원 | `inference/` 패키지, predictor registry, batch inference, modality별 input adapter | 14 |
| 14 | Evaluation architecture | 부분 | `metrics.py`, `evaluate.py`, `analysis/collector.py` | evaluator registry, predictions CSV, metric namespace, aggregation 강화 | 14, 17 |
| 15 | Sweep 설계 | 부분 | `experiment/sweep.py`, `sweep.py`, grid/manual, skip-completed | random sweep, parallel execution, retry, richer trial manifest | 15 |
| 16 | CLI 설계 | 부분 | `train.py`, `evaluate.py`, `infer.py`, `sweep.py`, `collect_results.py`, `validate_config.py` | `wafer_repro.cli.*` 계층화, 공통 CLI UX, compare/report CLI | 22 |
| 17 | 현재 코드 기준 마이그레이션 계획 | 완료 | Phase 0-6 구현 완료, Phase 7-10 hardening 진행 | 없음 | - |
| 18 | WM-811K 실험 축 예시 | 부분 | smoke configs, paper config, `wm811k_smoke_grid.yaml` | normalization 축, model x preprocess x training controls 대형 sweep 예시 | 17 |
| 19 | 논문 재현 baseline YAML | 완료 | `configs/experiments/wm811k/001_paper_reproduction.yaml` | 실제 full dataset 재현 실행은 사용자 데이터 필요 | - |
| 20 | 전처리 x 모델 x early stopping grid 예시 | 부분 | `configs/sweeps/wm811k_smoke_grid.yaml`, scheduler/early stopping 구현 | full grid config 추가, model axis와 early stopping axis 포함 | 17 |
| 21 | 코드별 상세 리팩터링 설명 | 부분 | `data.py` facade, `experiment/runner.py`, registry 도입 | `metrics.py`, `evaluate.py`, `infer.py`, `benchmark.py`, `run_experiments.py` 추가 정리 | 14, 22 |
| 22 | 고정 파라미터 기능 설계 | 부분 | `fixed.controls`, predefined split files, split CSV 저장 | frozen split CLI, dataset hash/size/mtime 저장, fixed override 정책 강화 | 13 |
| 23 | 시계열 데이터 확장 | 미구현 | 없음 | `timeseries_window` DataModule, forecasting task/model smoke | 18 |
| 24 | 이미지 폴더 데이터 확장 | 부분 | `datasets/image_folder/datamodule.py`, `configs/experiments/image_folder/000_smoke.yaml` | predefined split, normalization, richer augmentation, batch inference | 14, 19 |
| 25 | 비지도 이상탐지 | 미구현 | 없음 | anomaly detection task, reconstruction trainer/model, threshold/evaluator | 20 |
| 26 | 실험 비교 리포트 설계 | 부분 | `analysis/collector.py`, comparison CSV/grouped CSV | Markdown/HTML suite report, leaderboard, paired axis analysis, warnings | 17 |
| 27 | Validation과 오류 처리 | 부분 | `validate_config.py`, fixed control validation, runtime exceptions | registry-aware preflight validation, schema validation, path/split/monitor checks | 12 |
| 28 | 추가 의존성 제안 | 부분 | `PyYAML` 도입 | pydantic/rich/pyarrow/optuna 등은 필요 시 단계별 도입 | 12, 13, 15 |
| 29 | 구현 우선순위 체크리스트 | 부분 | 1-14번 대부분 반영, 15번 time series 미구현 | 새 Phase 11+ 로드맵으로 갱신 | 11 |
| 30 | 최종 목표 그림 | 부분 | `python -m wafer_repro.sweep`, registry 기반 데이터 추가 가능성 | `cli.sweep`, `cli.compare`, metric registry, 새 데이터셋/모델/지표 추가 규약 완성 | 22 |

## 완료된 Phase와 사양서 연결

| Phase | 구현 커밋 | 대응 사양서 장 |
|---|---|---|
| Phase 0 | `9f35019` | 1, 17 |
| Phase 1-3 | `5fa9de4` | 6, 8, 9, 10, 11, 12, 17, 19 |
| Phase 4 | `e0c3044` | 7, 12, 17, 21 |
| Phase 5 | `12c3ff1` | 15, 26, 17 |
| Phase 6 | `0bd5523` | 9.6, 17, 24 |
| Phase 7 | `72daf8b` | 13, 14, 24 |
| Phase 8 | `1f92a23` | 8.1, 9.1, 9.6 |
| Phase 9 | `7b4c12e` | 12.2, 12.3, 18.5, 20 |
| Phase 10 | `60a3e90` | 7.1, 15, 26 |
| Phase 11 | 현재 문서 | 1-30 전체 추적 체계 |
| Phase 12 | validation preflight 구현 | 6, 15, 27, 28 |
| Phase 13 | artifact contract 구현 | 7, 9, 11, 22 |
| Phase 14 | evaluation/inference architecture 분리 | 13, 14, 21, 24 |
| Phase 15 | sweep execution 고도화 | 12.4, 12.5, 15 |
| Phase 16 | multi-seed/k-fold suite 집계 | 12.4, 12.5, 14.2 |

## 후속 Phase 재정의

아래 Phase는 사양서 전체 커버리지를 완료하기 위한 새 기준이다. 이후 작업은 이 순서로 진행한다.

| Phase | 목표 | 주 대응 장 | 완료 기준 |
|---:|---|---|---|
| 11 | 사양서 전체 coverage matrix 작성 | 1-30 | 이 문서 작성, 남은 Phase 재정의, 커밋 |
| 12 | config schema validation과 preflight 오류 처리 강화 | 6, 27, 28 | registry/path/fixed/monitor/split 오류를 실행 전 탐지 |
| 13 | artifact contract 표준화 | 7, 9, 11, 22 | dataset hash, split hash, preprocessing manifest, predictions artifact |
| 14 | evaluation/inference architecture registry화 | 13, 14, 21, 24 | evaluator/predictor 구조 분리, WM-811K/image-folder 회귀 검증 |
| 15 | sweep execution 고도화 | 12.4, 12.5, 15 | random sweep, parallel, retry, stronger resume |
| 16 | multi-seed/k-fold suite 자동 실행과 집계 | 12.4, 12.5, 14.2 | seed/fold repeat config, grouped metric summary |
| 17 | suite report와 leaderboard 생성 | 14.2, 14.3, 18, 20, 26 | Markdown report, leaderboard, paired axis analysis |
| 18 | time-series modality 1차 구현 | 9.6, 10.5, 23 | toy time-series dataset, DataModule, forecasting smoke |
| 19 | tabular/image-folder 확장 hardening | 9.6, 24 | image-folder predefined split/normalization, tabular CSV smoke |
| 20 | anomaly detection task 1차 구현 | 10.4, 25 | autoencoder/reconstruction trainer smoke, threshold evaluation |
| 21 | regression/forecasting task 확장 | 10.3, 10.5, 23 | regression metrics, forecasting model/task smoke |
| 22 | package/CLI/test/compatibility 정리 | 5, 8, 16, 21, 30 | `cli` 계층, compatibility facade, smoke test scripts |
| 23 | 최종 문서와 사용 흐름 정리 | 29, 30 | README, examples, end-to-end commands, final checklist |

## 우선순위 원칙

1. 더 많은 task와 modality를 추가하기 전에 config validation과 artifact contract를 먼저 단단하게 만든다.
2. 비교 실험의 신뢰도를 위해 split, seed, dataset identity, metric namespace를 먼저 고정한다.
3. 새 modality는 최소 toy dataset과 smoke config를 함께 추가한다.
4. 각 Phase는 코드, 검증, 구현 로그, 커밋을 하나의 완료 단위로 남긴다.
5. 원격 push는 사용자가 요청했을 때 수행한다.

## 다음 작업

다음 Phase는 Phase 17이다.

Phase 17에서는 suite report와 leaderboard 생성을 구현한다. 최소 구현 범위는 다음이다.

- Markdown suite report
- leaderboard table
- per-axis method summary
- paired axis comparison
- warnings section
