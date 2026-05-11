param(
    [string]$Data = "..\LSWMD.pkl",
    [string]$Device = "auto",
    [int]$Epochs = 30,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m wafer_repro.run_experiments `
  --data $Data `
  --out-dir outputs/paper_runs `
  --models paper `
  --folds 4 `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --image-size 224 `
  --target-defect-count 10000 `
  --device $Device `
  --num-workers $NumWorkers

python -m wafer_repro.collect_results --runs-dir outputs/paper_runs --out outputs/comparison_summary.csv

