# Phase 17 Implementation Log

## 2026-05-20 - Suite report and leaderboard generation

Phase 17 adds human-readable Markdown reports on top of comparison CSV and JSON
summary artifacts.

## Implementation scope

- `src/wafer_repro/analysis/report.py`
  - Added Markdown suite report generation.
  - Added leaderboard table.
  - Added grouped metric table.
  - Added per-axis method summary.
  - Added paired axis macro-F1 comparison.
  - Added warnings section.

- `src/wafer_repro/analysis/collector.py`
  - `write_comparison` now also writes `<out>_report.md`.

## Generated comparison artifacts

For an output path such as `comparison.csv`, collector now writes:

```text
comparison.csv
comparison_grouped.csv
comparison_summary.json
comparison_report.md
```

## Verification

Syntax check:

```powershell
python -m py_compile `
  src\wafer_repro\analysis\report.py `
  src\wafer_repro\analysis\collector.py
```

Result: passed.

Seed/fold report:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments\wm811k_smoke_seed_fold `
  --out outputs\experiments\wm811k_smoke_seed_fold\comparison_phase17.csv
```

Result: wrote `comparison_phase17_report.md` with summary, leaderboard,
grouped results, per-axis analysis, and warnings.

Paired axis report:

```powershell
$env:PYTHONPATH='src'
python -m wafer_repro.collect_results `
  --runs-dir outputs\experiments\wm811k_smoke_grid `
  --out outputs\experiments\wm811k_smoke_grid\comparison_phase17.csv
```

Result: `comparison_phase17_report.md` includes paired preprocessing analysis:

```text
preprocessing | colormap | replicate | mean_delta_macro_f1 = 0.0023 | n_pairs = 1
```

## Remaining work

- Add richer report styling or HTML output later if needed.
- Add statistical tests only after enough repeated trials are available.
