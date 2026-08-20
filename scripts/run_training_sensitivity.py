from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.training_sensitivity import (  # noqa: E402
    SensitivityConfig,
    run_sensitivity_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the full 80/10/10 split, tuning, fitting, and assessment pipeline "
            "for multiple data-split seeds."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/training_sensitivity.example.json",
        help=(
            "Sensitivity configuration JSON. "
            "Default: configs/training_sensitivity.example.json"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one split with two folds and one trial per classifier.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed runs instead of resuming matching results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = SensitivityConfig.from_json(config_path)

    if args.force:
        config.resume_completed_runs = False
    if args.smoke_test:
        overrides = dict(config.training_overrides or {})
        overrides.update(
            {
                "tuning_cv_seeds": [42],
                "cv_splits": 2,
                "pixel_cv_splits": 2,
                "n_trials_dt": 1,
                "n_trials_rf": 1,
                "n_trials_xgb": 1,
                "top_candidates_per_run": 1,
                "xgb_max_estimators": 100,
                "xgb_early_stopping_rounds": 10,
            }
        )
        config = replace(
            config,
            output_dir=f"{config.output_dir}_smoke",
            split_seeds=config.split_seeds[:1],
            training_overrides=overrides,
            # Smoke settings are commonly edited between checks; recompute them.
            resume_completed_runs=False,
        )

    result = run_sensitivity_pipeline(config, project_root=ROOT)
    print(result["summary"].to_string(index=False))
    print(f"\nPer-run comparison: {result['paths']['results']}")
    print(f"Across-run summary: {result['paths']['summary']}")


if __name__ == "__main__":
    main()
