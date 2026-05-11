# Paper Reproduction Notes

대상 논문: Eunmi Shin, Chang D. Yoo, **Efficient Convolutional Neural Networks for Semiconductor Wafer Bin Map Classification**, Sensors 2023, 23, 1926.

이 프로젝트의 목적은 논문에 공개된 실험 조건을 기준으로 WM-811K wafer bin map 분류 파이프라인을 재현 가능한 코드로 정리하는 것입니다. 논문 원저자 코드가 포함된 재현은 아니므로, 모델 구현은 PyTorch/Torchvision의 표준 구현을 사용하고, 논문에 명시되지 않은 augmentation 세부 확률은 코드 옵션으로 노출했습니다.

## 1. 논문 실험 요약

논문은 wafer test 결과를 wafer bin map 이미지로 보고, 아래 9개 클래스를 분류합니다.

| Class | 논문 전체 labeled 샘플 수 | 비율 |
|---|---:|---:|
| Center | 4,294 | 2.48% |
| Donut | 555 | 0.32% |
| Edge-loc | 5,189 | 3.00% |
| Edge-ring | 9,680 | 5.60% |
| Loc | 3,593 | 2.08% |
| Near-full | 149 | 0.09% |
| Random | 866 | 0.50% |
| Scratch | 1,193 | 0.69% |
| None | 147,431 | 85.24% |
| Total | 172,950 | 100.00% |

주요 조건은 다음과 같습니다.

- 원본 데이터: WM-811K, 총 811,457 wafer
- labeled data: 172,950개
- unlabeled data: 639,507개
- 입력 이미지: 3채널 `224 x 224`
- 테스트셋: 전체 labeled data의 20%
- 나머지 80%를 4-fold로 나누어 training/validation 수행
- 실험 비율: training:validation:testing = 6:2:2
- 결함 8개 클래스만 augmentation으로 training data를 클래스당 10,000장까지 증가
- `None` 클래스는 augmentation하지 않고 자연 분포 유지
- batch size: 128
- optimizer: Adam
- learning rate: `1e-4`
- loss: cross entropy
- 주요 평가지표: accuracy, precision, recall, macro F1
- 효율 지표: parameters, memory, MAdds, FLOPs, training throughput, inference throughput

논문 reported test 결과는 대략 다음과 같습니다.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet18 | 0.980 | 0.915 | 0.877 | 0.895 |
| EfficientNetV2 | 0.979 | 0.918 | 0.868 | 0.885 |
| ShuffleNetV2 | 0.980 | 0.929 | 0.871 | 0.892 |
| ShuffleNetV2 0.5x | 0.979 | 0.920 | 0.858 | 0.884 |
| MobileNetV2 | 0.979 | 0.873 | 0.907 | 0.886 |
| MobileNetV3 | 0.980 | 0.909 | 0.885 | 0.895 |
| CNN-WDI | 0.979 | 0.938 | 0.861 | 0.895 |

## 2. 데이터 파이프라인

코드 위치: `src/wafer_repro/data.py`

### 2.1 LSWMD.pkl 로드

`load_lswmd()`는 pandas pickle로 `LSWMD.pkl`을 읽습니다. Kaggle WM-811K pickle은 보통 다음 컬럼을 포함합니다.

- `waferMap`: 2D numpy array. 일반적으로 `0=wafer 밖/빈칩`, `1=정상 chip`, `2=defective chip`로 해석합니다.
- `failureType`: 사람이 붙인 defect label. 값이 빈 배열인 row는 unlabeled로 취급합니다.
- `trianTestLabel`: Kaggle 원본 split 표기. 논문은 자체 80/20 split을 사용하므로 기본 코드에서는 직접 사용하지 않습니다.

라벨은 종종 `np.array([['Center']], dtype=object)`처럼 중첩되어 있으므로 `scalarize()`로 문자열 하나로 평탄화합니다. 이후 `LABEL_ALIASES`로 `none`, `Edge-Loc`, `edge_loc` 같은 표기 흔들림을 논문 표기인 `None`, `Edge-loc` 등으로 정규화합니다.

### 2.2 labeled row 필터링

논문은 unlabeled data를 학습에 사용하지 않고, labeled 172,950개만 사용합니다. 따라서 `failure_label`이 9개 논문 클래스에 속하는 row만 남깁니다.

코드는 `original_index`를 보존합니다. 이는 추론 시 Kaggle 원본 dataframe index로 wafer를 다시 찾기 위해 필요합니다. 필터링 후에는 `labeled_index`를 새로 부여합니다.

### 2.3 이미지 변환

논문은 wafer map을 `224 x 224 x 3` 이미지로 변환했다고 설명합니다. 이 프로젝트의 기본값은 `--channel-mode colormap`입니다.

`colormap` 변환:

- `0`: black
- `1`: green 계열
- `2`: yellow 계열

이는 논문 Figure 1의 “green normal, yellow defective” 표현과 잘 맞습니다. 다만 실제 학습에서 색 정보보다 공간 패턴이 핵심이라면 `--channel-mode replicate`도 사용할 수 있습니다. 이 모드는 `0/1/2` 값을 `0.0/0.5/1.0`로 정규화한 뒤 3채널로 복제합니다.

### 2.4 split 정책

논문 흐름:

1. labeled data 100%에서 test 20%를 stratified holdout으로 분리
2. 남은 80%를 4-fold로 분리
3. 각 fold에서 3개 fold를 train, 1개 fold를 validation으로 사용
4. 비율은 train 60%, validation 20%, test 20%

코드는 두 모드를 제공합니다.

- 기본 single run: `make_single_split()`으로 6:2:2를 한 번 만듭니다.
- 논문식 k-fold: `--fold 0`, `--fold 1`, `--fold 2`, `--fold 3`로 각 fold를 실행합니다.

전체 비교를 자동화하려면:

```powershell
python -m wafer_repro.run_experiments --data ..\LSWMD.pkl --models paper --folds 4
```

### 2.5 augmentation 정책

논문은 결함 8개 클래스 training data를 augmentation으로 클래스당 10,000장까지 늘리고, `None` 클래스는 유지합니다. 논문 Table 7의 80% training 기준 개수는 다음과 같습니다.

| Class | Original training | Augmented training | Testing |
|---|---:|---:|---:|
| Center | 3,435 | 10,000 | 859 |
| Donut | 444 | 10,000 | 111 |
| Edge-loc | 4,151 | 10,000 | 1,038 |
| Edge-ring | 7,744 | 10,000 | 1,936 |
| Loc | 2,874 | 10,000 | 719 |
| Near-full | 119 | 10,000 | 30 |
| Random | 693 | 10,000 | 173 |
| Scratch | 954 | 10,000 | 239 |
| None | 117,945 | 117,945 | 29,486 |
| Total | 138,360 | 197,945 | 34,590 |

이 프로젝트는 data leakage를 피하기 위해 split을 먼저 만들고, train split에만 augmentation record를 추가합니다. 즉 validation/test에는 원본 분포만 들어갑니다.

적용되는 stochastic transform:

- `Resize`
- `RandomCrop`
- `RandomRotation`
- `RandomErasing`
- `GaussianBlur`
- `RandomHorizontalFlip`
- `RandomVerticalFlip`

논문은 각 transform의 확률과 파라미터를 자세히 공개하지 않았으므로, 코드에서는 다음 기본값을 사용합니다.

- `--rotation-degrees 180`
- `--crop-padding 16`
- `--blur-prob 0.2`
- `--erase-prob 0.25`

재현 실험에서 이 값들은 튜닝 대상입니다. 완전히 같은 metric을 맞추려면 seed, augmentation 강도, 학습 epoch, pretrained 여부, hardware backend까지 함께 고정해야 합니다.

## 3. 모델 설명

코드 위치: `src/wafer_repro/models.py`

### 3.1 ResNet18

논문은 ResNet18을 널리 알려진 image classification baseline으로 사용합니다. 잔차 연결을 통해 깊은 네트워크의 gradient 흐름을 안정화합니다. 이 프로젝트에서는 `torchvision.models.resnet18`의 마지막 `fc`만 9-class classifier로 교체합니다.

### 3.2 ShuffleNetV2 1.0x / 0.5x

ShuffleNetV2는 mobile inference를 목표로 설계된 경량 CNN입니다.

핵심 아이디어:

- channel split으로 feature 일부만 변환하고 나머지는 shortcut처럼 유지
- channel shuffle로 group 간 정보 흐름 확보
- 지나친 group convolution과 fragmentation을 줄여 실제 speed를 개선

논문은 1.0x와 0.5x를 비교합니다. 0.5x는 parameter와 FLOPs가 더 작지만, 논문 결과에서는 F1이 상대적으로 낮았습니다.

### 3.3 MobileNetV2

MobileNetV2는 depthwise separable convolution에 inverted residual과 linear bottleneck을 결합합니다.

핵심 아이디어:

- depthwise convolution으로 공간 필터 비용 감소
- pointwise convolution으로 channel mixing
- expansion factor `t=6` 계열 bottleneck 구조
- 마지막 bottleneck projection에는 비선형 활성화를 두지 않아 정보 손실 완화

논문 결과에서는 recall이 높은 편이지만 CPU inference가 MobileNetV3보다 느렸습니다.

### 3.4 MobileNetV3-Small

논문이 가장 좋은 trade-off로 결론 낸 모델입니다.

핵심 아이디어:

- NAS/NetAdapt 기반으로 mobile latency를 고려한 구조 탐색
- h-swish 활성화로 swish의 계산 부담 완화
- squeeze-and-excitation으로 channel-wise feature recalibration
- small variant는 wafer map처럼 비교적 단순한 패턴 분류에서 충분한 성능을 내면서 parameter와 FLOPs가 낮음

논문 reported 결과:

- accuracy 약 0.980
- macro F1 약 0.895
- ResNet18 대비 parameter 약 7.5배 감소
- ResNet18 대비 training speed 약 7.2배, inference speed 약 4.9배 향상

### 3.5 EfficientNetV2-S

EfficientNetV2는 training speed와 parameter efficiency를 개선한 EfficientNet 계열입니다.

핵심 아이디어:

- 초반 layer에서 Fused-MBConv 사용
- 후반 layer에서 MBConv 사용
- network scaling을 조정해 학습 속도와 정확도 균형 확보

논문에서는 경량 모델로 비교되지만, ResNet18보다 parameter와 memory가 더 클 수 있습니다.

### 3.6 CNN-WDI-style CNN

논문은 관련 연구 CNN-WDI를 비교표에 포함하지만, Sensors 논문 안에 완전한 재구현 구조가 제공되지는 않습니다. 따라서 이 프로젝트의 `cnn_wdi`는 “직접적인 mobile backbone 기법 없이 stacked convolution block으로 분류하는 compact CNN” 역할의 비교 baseline입니다.

정확히 CNN-WDI 논문 구조를 재현하려면 해당 원논문의 feature branch와 classifier 구성을 별도로 반영해야 합니다. 이 프로젝트에서는 논문 비교 실험을 한 번에 돌릴 수 있도록 실용적인 CNN baseline을 제공합니다.

## 4. 학습 루프

코드 위치: `src/wafer_repro/train.py`

한 epoch에서 수행하는 일:

1. dataloader가 wafer map을 이미지 tensor로 변환
2. 모델 forward
3. cross entropy loss 계산
4. Adam optimizer로 parameter update
5. train accuracy와 macro F1 계산
6. validation split에서 loss, accuracy, macro F1 계산
7. validation macro F1이 가장 좋으면 `best.pt` 저장

논문은 imbalanced dataset이므로 accuracy만 보면 `None` 클래스에 치우친 성능을 과대평가할 수 있다고 설명합니다. 따라서 코드도 checkpoint 선택 기준을 validation macro F1로 둡니다.

## 5. 평가 산출물

코드 위치: `src/wafer_repro/metrics.py`

평가 시 저장되는 파일:

- `classification_report.csv/json`
- `confusion_matrix.csv/png`
- `confusion_matrix_normalized.csv/png`
- `summary.json`

`summary.json`에는 다음 값이 들어갑니다.

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1

논문 Table 10과 비교할 때는 macro precision/recall/F1을 우선 확인하는 편이 좋습니다.

## 6. 구현상 차이와 주의점

1. 논문 원본 augmentation 파라미터가 완전히 공개되어 있지 않습니다.
   이 프로젝트는 같은 transform family를 사용하지만 확률과 강도는 합리적인 기본값으로 지정했습니다.

2. 논문은 “80% training data를 augmentation 후 4-fold”로 읽힐 여지가 있습니다.
   이 구현은 validation leakage를 피하기 위해 train fold에만 augmentation을 적용합니다.

3. Torchvision 표준 구현을 사용합니다.
   논문 Table 3-6의 구조와 같은 계열 모델이지만, 세부 layer 구현은 torchvision 버전에 의존합니다.

4. pretrained 기본값은 꺼져 있습니다.
   논문은 ImageNet pretrained 사용 여부를 명확히 강조하지 않으므로 `--pretrained`는 옵션으로 두었습니다.

5. 실험 시간은 깁니다.
   224x224 이미지, 197,945개 수준의 augmented train records, 여러 큰 모델을 4-fold로 학습하면 GPU가 있어도 시간이 꽤 걸립니다.

## 7. 추천 실행 순서

1. toy data로 코드 경로 확인
2. 실제 데이터에서 `mobilenet_v3_small` 단일 run
3. `shufflenet_v2_x1_0`, `mobilenet_v2`, `resnet18` 순서로 비교 확대
4. 전체 4-fold paper run
5. benchmark로 parameter/FLOPs/throughput 비교

예시:

```powershell
python -m wafer_repro.train --data ..\LSWMD.pkl --model mobilenet_v3_small --epochs 30 --batch-size 128
python -m wafer_repro.benchmark --models paper --device auto
python -m wafer_repro.collect_results --runs-dir outputs/paper_runs --out outputs/comparison_summary.csv
```

