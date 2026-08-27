from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.leave_one_site_out import (  # noqa: E402
    LeaveOneSiteOutConfig,
    run_leave_one_site_out_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hold out each lake in turn, run one five-fold hyperparameter search per "
            "classifier, and assess DT, RF, and XGBoost robustness."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/leave_one_site_out.example.json",
        help=(
            "LOSO configuration JSON. "
            "Default: configs/leave_one_site_out.example.json"
        ),
    )
    parser.add_argument(
        "--held-out-site",
        action="append",
        help="Run only this site; repeat the option for multiple sites.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one held-out site with two folds and one trial per classifier.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute matching completed site runs instead of resuming them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = LeaveOneSiteOutConfig.from_json(config_path)

    if args.held_out_site:
        config.held_out_sites = args.held_out_site
    if args.force:
        config.resume_completed_runs = False
    if args.smoke_test:
        sites = config.held_out_sites or ["ageri"]
        config = replace(
            config,
            output_dir=f"{config.output_dir}_smoke",
            held_out_sites=sites[:1],
            cv_splits=2,
            n_trials_dt=1,
            n_trials_rf=1,
            n_trials_xgb=1,
            xgb_max_estimators=100,
            xgb_early_stopping_rounds=10,
            resume_completed_runs=False,
        )

    result = run_leave_one_site_out_pipeline(config, project_root=ROOT)
    print(result["summary"].to_string(index=False))
    print(f"\nPer-site results: {result['paths']['results']}")
    print(f"Across-site summary: {result['paths']['summary']}")


if __name__ == "__main__":
    main()
