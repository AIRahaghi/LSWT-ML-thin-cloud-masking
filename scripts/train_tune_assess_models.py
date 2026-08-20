from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.model_training import TrainingConfig, run_training_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and assess DT, RF, and XGBoost thin-cloud classifiers."
    )
    parser.add_argument("--config", help="JSON config file.")
    parser.add_argument("--train-csv", help="Training CSV (80 percent split).")
    parser.add_argument("--test-csv", help="Familiar-scene test CSV (10 percent split).")
    parser.add_argument(
        "--los-test-csv",
        help="Leave-out-of-scene test CSV (10 percent); scene IDs must be absent elsewhere.",
    )
    parser.add_argument("--output-dir", help="Folder for models and reports.")
    parser.add_argument("--label-column", default=None, help="Label column. Default: lst_class.")
    parser.add_argument("--cv-splits", type=int, default=None, help="Stratified CV folds.")
    parser.add_argument("--scoring", default=None, help="scikit-learn scoring metric.")
    parser.add_argument(
        "--n-trials-dt",
        type=int,
        default=None,
        help="Decision Tree Optuna trials per tuning CV seed.",
    )
    parser.add_argument(
        "--n-trials-rf",
        type=int,
        default=None,
        help="Random Forest Optuna trials per tuning CV seed.",
    )
    parser.add_argument(
        "--n-trials-xgb",
        type=int,
        default=None,
        help="XGBoost Optuna trials per tuning CV seed.",
    )
    parser.add_argument("--no-tuning", action="store_true", help="Fit default models without Optuna.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        if not args.train_csv or not args.test_csv or not args.output_dir:
            raise SystemExit("--train-csv, --test-csv, and --output-dir are required without --config.")
        config = TrainingConfig(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
        )

    for arg_name, field_name in [
        ("train_csv", "train_csv"),
        ("test_csv", "test_csv"),
        ("los_test_csv", "los_test_csv"),
        ("output_dir", "output_dir"),
        ("label_column", "label_column"),
        ("cv_splits", "cv_splits"),
        ("scoring", "scoring"),
        ("n_trials_dt", "n_trials_dt"),
        ("n_trials_rf", "n_trials_rf"),
        ("n_trials_xgb", "n_trials_xgb"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)

    if args.no_tuning:
        config.n_trials_dt = 0
        config.n_trials_rf = 0
        config.n_trials_xgb = 0

    result = run_training_pipeline(config)
    summary = {
        name: {
            "train_accuracy": metrics["datasets"]["train"]["accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "los_test_accuracy": metrics.get("los_test_accuracy"),
            "test_balanced_accuracy": metrics["test_balanced_accuracy"],
            "los_test_balanced_accuracy": metrics.get("los_test_balanced_accuracy"),
            "scene_grouped_cv_accuracy_mean": metrics["cross_validation"]
            .get("scene_grouped", {})
            .get("accuracy", {})
            .get("mean"),
            "scene_grouped_cv_overall_accuracy": metrics["cross_validation"]
            .get("scene_grouped", {})
            .get("accuracy", {})
            .get("pooled"),
            "pixel_stratified_cv_accuracy_mean": metrics["cross_validation"]
            .get("pixel_stratified", {})
            .get("accuracy", {})
            .get("mean"),
            "pixel_stratified_cv_overall_accuracy": metrics["cross_validation"]
            .get("pixel_stratified", {})
            .get("accuracy", {})
            .get("pooled"),
            "selection_repeated_grouped_mean": result["metadata"]["model_selection"][
                name
            ].get("selected_mean_score"),
            "selection_repeated_grouped_std": result["metadata"]["model_selection"][
                name
            ].get("selected_std_score"),
            "selection_stability_score": result["metadata"]["model_selection"][
                name
            ].get("selected_stability_score"),
        }
        for name, metrics in result["metrics"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
