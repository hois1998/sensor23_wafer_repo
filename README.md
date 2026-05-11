# Sensors 2023 Wafer Bin Map Reproduction

이 폴더는 Shin & Yoo, **“Efficient Convolutional Neural Networks for Semiconductor Wafer Bin Map Classification”**, Sensors 2023, 23, 1926 논문을 최대한 모사해 WM-811K/LSWMD wafer map 분류를 학습, 추론, 평가, 모델 비교까지 수행하도록 만든 PyTorch 프로젝트입니다.

원본 데이터는 이미 상위 폴더에 있는 `..\LSWMD.pkl`을 기본 경로로 사용합니다.

## 1. 설치

```powershell
cd "C:\Users\youngho\Repositories\새 폴더\sensors23_wafer_repro"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

FLOPs/MAdds까지 재고 싶으면 아래처럼 profile extra를 설치합니다.

```powershell
pip install -e ".[profile]"
```

## 2. 빠른 학습 예시

논문 기본 추천 모델인 `MobileNetV3-Small`을 224x224, batch 128, Adam lr 1e-4로 학습합니다.

```powershell
python -m wafer_repro.train `
  --data ..\LSWMD.pkl `
  --model mobilenet_v3_small `
  --epochs 30 `
  --batch-size 128 `
  --device auto
```

결과는 `outputs/runs/<run-name>` 아래에 저장됩니다.

- `best.pt`, `last.pt`: 체크포인트
- `config.json`: 실행 옵션
- `data_summary.json`: split/증강 후 클래스 분포
- `history.csv`: epoch별 train/validation loss, accuracy, macro F1
- `splits/*.csv`: train/validation/test row 목록
- `metrics/*`: classification report, confusion matrix, normalized confusion matrix
- `test_summary.json`: 테스트셋 accuracy, macro precision/recall/F1

## 3. 논문 비교 모델 전체 실행

논문 실험군을 4-fold로 순차 실행합니다. 오래 걸립니다.

```powershell
.\scripts\paper_full_run.ps1 -Data "..\LSWMD.pkl" -Device auto -Epochs 30 -BatchSize 128
```

포함 모델:

- `resnet18`
- `efficientnet_v2_s`
- `shufflenet_v2_x1_0`
- `shufflenet_v2_x0_5`
- `mobilenet_v2`
- `mobilenet_v3_small`
- `cnn_wdi`

실행 후 `outputs/comparison_summary.csv`와 `outputs/comparison_summary_by_model.csv`가 생성됩니다.

## 4. 평가와 추론

저장된 체크포인트를 다시 평가:

```powershell
python -m wafer_repro.evaluate `
  --data ..\LSWMD.pkl `
  --checkpoint outputs\runs\mobilenet_v3_small_YYYYMMDD_HHMMSS\best.pt `
  --split test
```

LSWMD의 labeled dataframe row 하나를 추론:

```powershell
python -m wafer_repro.infer `
  --data ..\LSWMD.pkl `
  --checkpoint outputs\runs\mobilenet_v3_small_YYYYMMDD_HHMMSS\best.pt `
  --row-index 0 `
  --out-dir outputs\single_infer
```

## 5. 벤치마크

모델별 parameter 수, 선택적으로 MAdds/MFLOPs, 학습/추론 throughput을 측정합니다.

```powershell
python -m wafer_repro.benchmark --models paper --device auto --batch-size 128 --image-size 224
```

`thop`이 설치되어 있으면 MAdds/MFLOPs가 채워지고, 없으면 해당 칸은 비어 있습니다.

## 6. Smoke Test

현재 환경에 `torch/torchvision` 설치 전이라도 toy pickle 생성은 가능합니다.

```powershell
python .\scripts\make_toy_lswmd.py --out data\toy_LSWMD.pkl --per-class 12
```

학습까지 확인하려면 torch 설치 후:

```powershell
python -m wafer_repro.train `
  --data data\toy_LSWMD.pkl `
  --model small_cnn `
  --epochs 1 `
  --batch-size 16 `
  --image-size 64 `
  --target-defect-count 20 `
  --max-samples-per-class 12 `
  --device cpu
```

## 7. 상세 문서

데이터 파이프라인과 모델 설명은 [docs/paper_reproduction_notes.md](docs/paper_reproduction_notes.md)에 길게 정리했습니다. 특히 논문 수치, augmentation, split 정책, 모델별 구조 의미, 구현 차이를 그 문서에 적어두었습니다.

