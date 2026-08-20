"""Repeated-holdout sensitivity analysis for the model-training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data_splitting import SplitConfig, run_split_pipeline
from .model_training import TrainingConfig, run_training_pipeline


MODEL_FILENAMES = {
    "decision_tree": "DT_best_all_general.pkl",
    "random_forest": "RF_best_all_general.pkl",
    "xgboost": "XGB_best_all_general.pkl",
}

GENERATED_PATH_FIELDS = {"train_csv", "test_csv", "los_test_csv", "output_dir"}


@dataclass
class SensitivityConfig:
    """Configuration for repeated 80/10/10 split-and-tune experiments."""

    split_config: str
    training_config: str
    output_dir: str
    split_seeds: list[int]
    training_overrides: dict[str, Any] | None = None
    resume_completed_runs: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "SensitivityConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


def run_sensitivity_pipeline(
    config: SensitivityConfig,
    *,
    project_root: str | Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create, train, and compare one model set per requested split seed."""

    root = Path(project_root).resolve()
    _validate_sensitivity_config(config)
    output_dir = _resolve_path(root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_split = SplitConfig.from_json(_resolve_path(root, config.split_config))
    try:
        base_training = TrainingConfig.from_json(
            _resolve_path(root, config.training_config)
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        raise RuntimeError(
            "The in-memory TrainingConfig class does not match the training JSON. "
            "Restart the Jupyter kernel and rerun the notebook from its first cell."
        ) from exc
    base_split.input_csvs = [
        str(_resolve_path(root, path)) for path in base_split.input_csvs
    ]
    training_overrides = config.training_overrides or {}
    unknown_overrides = sorted(set(training_overrides) - set(asdict(base_training)))
    if unknown_overrides:
        raise ValueError(f"Unknown training overrides: {unknown_overrides}")
    for field_name, value in training_overrides.items():
        setattr(base_training, field_name, value)

    all_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    los_scenes_by_run: dict[str, set[str]] = {}
    for run_index, split_seed in enumerate(config.split_seeds, start=1):
        run_name = f"run_{run_index:02d}_seed_{split_seed}"
        dataset_dir = output_dir / "datasets" / run_name
        model_dir = output_dir / "models" / run_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        split_config = replace(
            base_split,
            train_csv=str(dataset_dir / "df_train_80.csv"),
            test_csv=str(dataset_dir / "df_test_inscene_10.csv"),
            los_test_csv=str(dataset_dir / "df_test_los_10.csv"),
            manifest_csv=str(dataset_dir / "scene_split_manifest.csv"),
            summary_json=str(dataset_dir / "scene_split_summary.json"),
            random_state=split_seed,
        )
        training_config = replace(
            base_training,
            train_csv=split_config.train_csv,
            test_csv=split_config.test_csv,
            los_test_csv=split_config.los_test_csv,
            output_dir=str(model_dir),
        )
        _assert_generated_run_paths(
            split_config,
            training_config,
            dataset_dir,
            model_dir,
        )
        fingerprint = _run_fingerprint(split_config, training_config)
        run_summary_path = model_dir / "sensitivity_run_summary.json"

        saved = _load_completed_run(
            run_summary_path,
            fingerprint,
            model_dir,
        ) if config.resume_completed_runs else None
        if saved is not None:
            progress(f"[{run_index}/{len(config.split_seeds)}] Reusing {run_name}.")
            result_rows = saved["results"]
            split_summary = saved["split_summary"]
        else:
            progress(
                f"[{run_index}/{len(config.split_seeds)}] Creating split {run_name}."
            )
            split_result = run_split_pipeline(split_config)
            split_summary = split_result["summary"]
            _write_json(dataset_dir / "effective_split_config.json", asdict(split_config))

            generated_paths = [
                Path(split_config.train_csv),
                Path(split_config.test_csv),
                Path(split_config.los_test_csv),
            ]
            missing_generated = [str(path) for path in generated_paths if not path.is_file()]
            if missing_generated:
                raise RuntimeError(
                    "Split generation did not create all required training datasets: "
                    f"{missing_generated}"
                )

            progress(
                f"[{run_index}/{len(config.split_seeds)}] Tuning and evaluating {run_name}."
            )
            progress(f"  train_csv: {training_config.train_csv}")
            progress(f"  test_inscene_csv: {training_config.test_csv}")
            progress(f"  los_csv: {training_config.los_test_csv}")
            training_result = run_training_pipeline(training_config)
            _write_json(model_dir / "effective_training_config.json", asdict(training_config))
            result_rows = _result_rows(
                run_index,
                run_name,
                split_seed,
                training_result,
                split_config,
                model_dir,
            )
            pd.DataFrame(result_rows).to_csv(
                model_dir / "sensitivity_model_metrics.csv",
                index=False,
            )
            _write_json(
                run_summary_path,
                {
                    "fingerprint": fingerprint,
                    "run_index": run_index,
                    "run_name": run_name,
                    "split_seed": split_seed,
                    "split_summary": split_summary,
                    "results": result_rows,
                },
            )

        all_rows.extend(result_rows)
        los_scene_ids = set(split_summary["los_selection"]["scene_ids"])
        los_scenes_by_run[run_name] = los_scene_ids
        run_records.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "split_seed": split_seed,
                "dataset_dir": str(dataset_dir),
                "model_dir": str(model_dir),
                "los_scene_ids": sorted(los_scene_ids),
                "split_balance_score": split_summary["los_selection"]["balance_score"],
            }
        )

    results = pd.DataFrame(all_rows).sort_values(
        ["run_index", "model"], kind="stable"
    )
    comparison_path = output_dir / "sensitivity_results.csv"
    results.to_csv(comparison_path, index=False)

    summary = _summarize_balanced_accuracy(results)
    summary_path = output_dir / "sensitivity_summary_by_model.csv"
    summary.to_csv(summary_path, index=False)

    overlap = _los_overlap_table(los_scenes_by_run)
    overlap_path = output_dir / "los_scene_overlap.csv"
    overlap.to_csv(overlap_path, index=False)

    manifest = {
        "config": asdict(config),
        "project_root": str(root),
        "runs": run_records,
        "result_files": {
            "per_run_model_metrics": str(comparison_path),
            "balanced_accuracy_summary": str(summary_path),
            "los_scene_overlap": str(overlap_path),
        },
        "selection_note": (
            "Test and LOS scores are reported for sensitivity assessment only and are not "
            "used during hyperparameter tuning or within-run model selection."
        ),
    }
    manifest_path = output_dir / "sensitivity_manifest.json"
    _write_json(manifest_path, manifest)
    progress(f"Sensitivity comparison written to {comparison_path}.")
    return {
        "results": results,
        "summary": summary,
        "los_scene_overlap": overlap,
        "manifest": manifest,
        "paths": {
            "results": str(comparison_path),
            "summary": str(summary_path),
            "los_scene_overlap": str(overlap_path),
            "manifest": str(manifest_path),
        },
    }


def _result_rows(
    run_index: int,
    run_name: str,
    split_seed: int,
    training_result: dict[str, Any],
    split_config: SplitConfig,
    model_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in training_result["metrics"].items():
        datasets = metrics["datasets"]
        selection = training_result["metadata"]["model_selection"][model_name]
        grouped = metrics["cross_validation"].get("scene_grouped", {})
        pixel = metrics["cross_validation"].get("pixel_stratified", {})
        rows.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "split_seed": split_seed,
                "model": model_name,
                "train_accuracy": datasets["train"]["accuracy"],
                "train_balanced_accuracy": datasets["train"]["balanced_accuracy"],
                "train_f1_macro": datasets["train"]["f1_macro"],
                "test_inscene_accuracy": datasets["test"]["accuracy"],
                "test_inscene_balanced_accuracy": datasets["test"]["balanced_accuracy"],
                "test_inscene_f1_macro": datasets["test"]["f1_macro"],
                "los_accuracy": datasets["los_test"]["accuracy"],
                "los_balanced_accuracy": datasets["los_test"]["balanced_accuracy"],
                "los_f1_macro": datasets["los_test"]["f1_macro"],
                "grouped_cv_balanced_accuracy_mean": _nested(
                    grouped, "balanced_accuracy", "mean"
                ),
                "grouped_cv_balanced_accuracy_std": _nested(
                    grouped, "balanced_accuracy", "std"
                ),
                "grouped_cv_balanced_accuracy_pooled": _nested(
                    grouped, "balanced_accuracy", "pooled"
                ),
                "pixel_cv_balanced_accuracy_mean": _nested(
                    pixel, "balanced_accuracy", "mean"
                ),
                "selection_mean": selection.get("selected_mean_score"),
                "selection_std": selection.get("selected_std_score"),
                "selection_stability_score": selection.get(
                    "selected_stability_score"
                ),
                "final_n_estimators": selection.get("final_n_estimators"),
                "selected_parameters": json.dumps(
                    selection.get("parameters", {}), sort_keys=True
                ),
                "train_csv": split_config.train_csv,
                "test_inscene_csv": split_config.test_csv,
                "los_csv": split_config.los_test_csv,
                "model_path": str(model_dir / MODEL_FILENAMES[model_name]),
                "metrics_path": str(model_dir / "metrics_summary.json"),
                "metadata_path": str(model_dir / "model_metadata.json"),
            }
        )
    return rows


def _summarize_balanced_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_balanced_accuracy",
        "test_inscene_balanced_accuracy",
        "los_balanced_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in results.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model_name, "n_runs": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _los_overlap_table(los_scenes_by_run: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = list(los_scenes_by_run)
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            intersection = los_scenes_by_run[left] & los_scenes_by_run[right]
            union = los_scenes_by_run[left] | los_scenes_by_run[right]
            rows.append(
                {
                    "run_a": left,
                    "run_b": right,
                    "overlap_scene_count": len(intersection),
                    "union_scene_count": len(union),
                    "jaccard_similarity": len(intersection) / len(union) if union else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _load_completed_run(
    summary_path: Path,
    fingerprint: str,
    model_dir: Path,
) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing sensitivity run has a different configuration: {model_dir}. "
            "Use a new output_dir or disable resume_completed_runs to recompute it."
        )
    required = [
        model_dir / filename for filename in MODEL_FILENAMES.values()
    ] + [
        model_dir / "metrics_summary.json",
        model_dir / "model_metadata.json",
        model_dir / "sensitivity_model_metrics.csv",
    ]
    if not all(path.exists() for path in required):
        return None
    return saved


def _run_fingerprint(
    split_config: SplitConfig,
    training_config: TrainingConfig,
) -> str:
    input_hashes = {
        path: _sha256(Path(path)) for path in split_config.input_csvs
    }
    payload = {
        "split_config": asdict(split_config),
        "training_config": asdict(training_config),
        "input_sha256": input_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sensitivity_config(config: SensitivityConfig) -> None:
    if not config.split_seeds:
        raise ValueError("At least one split seed is required.")
    if len(config.split_seeds) != len(set(config.split_seeds)):
        raise ValueError("split_seeds must contain distinct values.")
    if any(not isinstance(seed, int) for seed in config.split_seeds):
        raise ValueError("Every split seed must be an integer.")
    forbidden_overrides = sorted(
        set(config.training_overrides or {}) & GENERATED_PATH_FIELDS
    )
    if forbidden_overrides:
        raise ValueError(
            "Sensitivity training paths are generated per run and cannot be overridden: "
            f"{forbidden_overrides}"
        )


def _assert_generated_run_paths(
    split_config: SplitConfig,
    training_config: TrainingConfig,
    dataset_dir: Path,
    model_dir: Path,
) -> None:
    expected_datasets = {
        "train_csv": dataset_dir / "df_train_80.csv",
        "test_csv": dataset_dir / "df_test_inscene_10.csv",
        "los_test_csv": dataset_dir / "df_test_los_10.csv",
    }
    for field_name, expected in expected_datasets.items():
        split_path = Path(getattr(split_config, field_name)).resolve()
        training_path = Path(getattr(training_config, field_name)).resolve()
        expected_path = expected.resolve()
        if split_path != expected_path or training_path != expected_path:
            raise RuntimeError(
                f"Sensitivity run '{field_name}' must use its generated dataset at "
                f"'{expected_path}', got split='{split_path}', training='{training_path}'."
            )
    if Path(training_config.output_dir).resolve() != model_dir.resolve():
        raise RuntimeError(
            "Sensitivity model output must use the run-specific model directory."
        )


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
