from __future__ import annotations

import argparse
import json

from wafer_repro.core.config import config_hash, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize an experiment YAML config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Override config value as KEY=VALUE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    summary = {
        "config": args.config,
        "schema_version": config.get("schema_version"),
        "experiment": config.get("experiment", {}).get("name"),
        "suite": config.get("experiment", {}).get("suite"),
        "hash": config_hash(config, exclude_paths={"runtime.output_dir", "runtime.run_dir", "runtime.config_hash"}),
        "status": "valid",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

