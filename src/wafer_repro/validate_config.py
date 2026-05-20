from __future__ import annotations

import argparse
import json
import sys

from wafer_repro.core.config import config_hash, load_config
from wafer_repro.core.validation import ConfigValidationError, validate_experiment_config, validate_sweep_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize an experiment or sweep YAML config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Override config value as KEY=VALUE.")
    parser.add_argument("--check-paths", action="store_true", help="Require configured local data/split paths to exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config, args.overrides)
        if "sweep" in config:
            validate_sweep_config(config)
            summary = {
                "config": args.config,
                "schema_version": config.get("schema_version"),
                "sweep": config.get("sweep", {}).get("name"),
                "hash": config_hash(config),
                "status": "valid",
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return

        validate_experiment_config(config, check_paths=args.check_paths)
    except ConfigValidationError as exc:
        summary = {
            "config": args.config,
            "status": "invalid",
            "errors": [issue.to_dict() for issue in exc.issues],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        sys.exit(1)
    except Exception as exc:
        summary = {
            "config": args.config,
            "status": "invalid",
            "errors": [{"path": "<config>", "message": str(exc)}],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        sys.exit(1)

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
