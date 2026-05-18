from __future__ import annotations

import argparse

from wafer_repro.experiment.sweep import run_sweep, write_trial_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or materialize an experiment sweep.")
    parser.add_argument("--config", required=True, help="Path to sweep YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Write trial configs without running them.")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip trials whose run_manifest.json already has status=completed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        suite_dir, trials = write_trial_configs(args.config)
        print(f"Wrote {len(trials)} trial configs to {suite_dir}")
        return
    suite_dir = run_sweep(args.config, skip_completed=args.skip_completed)
    print(f"Sweep completed: {suite_dir}")


if __name__ == "__main__":
    main()
