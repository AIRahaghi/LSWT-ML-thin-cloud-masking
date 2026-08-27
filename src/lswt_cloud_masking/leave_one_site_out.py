"""Fast leave-one-site-out tuning and robustness assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from .features import CLASS_LABELS, DEFAULT_DROP_COLUMNS


MODEL_FILENAMES = {
    "decision_tree": "DT_best_all_general.pkl",
    "random_forest": "RF_best_all_general.pkl",
    "xgboost": "XGB_best_all_general.pkl",
}

DATASET_FILENAMES = {
    "train": "df_train_80.csv",
    "test": "df_test_20.csv",
    "leave_out_site": "df_leave_out_site.csv",
}

SEASONS = ("DJF", "MAM", "JJA", "SON")


@dataclass
class LeaveOneSiteOutConfig:
    """Configuration for six independent leave-one-site-out experiments."""

    input_csvs: list[str]
    output_dir: str
    held_out_sites: list[str] | None = None
    site_column: str = "lakeN"
    scene_column: str = "scene_id"
    label_column: str = "lst_class"
    drop_columns: list[str] = field(default_factory=lambda: list(DEFAULT_DROP_COLUMNS))
    test_fraction: float = 0.2
    split_random_state: int = 42
    scene_split_search_iterations: int = 100_000
    scene_split_max_test_fraction_deviation: float = 0.01
    overlap_exception_scene_id: str | None = None
    overlap_exception_site: str | None = None
    overlap_exception_class: int | None = None
    overlap_exception_train_rows: int = 0
    overlap_exception_test_rows: int = 0
    cv_splits: int = 5
    cv_random_state: int = 42
    model_random_state: int = 42
    optuna_seed: int = 42
    scoring: str = "balanced_accuracy"
    n_trials_dt: int = 120
    n_trials_rf: int = 120
    n_trials_xgb: int = 120
    optuna_n_jobs: int = 1
    rf_n_jobs: int = -1
    xgb_n_jobs: int = -1
    xgb_sample_weight: str | None = "balanced"
    xgb_max_estimators: int = 1500
    xgb_early_stopping_rounds: int = 50
    resume_completed_runs: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "LeaveOneSiteOutConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


@dataclass
class PreparedSiteData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    x_leave_out_site: pd.DataFrame
    y_leave_out_site: pd.Series
    feature_columns: list[str]
    dropped_incomplete_rows: dict[str, int]


def run_leave_one_site_out_pipeline(
    config: LeaveOneSiteOutConfig,
    *,
    project_root: str | Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run one fast single-search training experiment per held-out site."""

    root = Path(project_root).resolve()
    config = _resolved_config(config, root)
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source, source_columns, input_records = _load_source_data(config)
    available_sites = sorted(source[config.site_column].astype(str).unique())
    held_out_sites = [str(site) for site in (config.held_out_sites or available_sites)]
    missing_sites = sorted(set(held_out_sites) - set(available_sites))
    if missing_sites:
        raise ValueError(f"Requested held-out sites were not found: {missing_sites}")

    all_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for run_index, held_out_site in enumerate(held_out_sites, start=1):
        run_name = f"held_out_{_slug(held_out_site)}"
        dataset_dir = output_dir / "datasets" / run_name
        model_dir = output_dir / "models" / run_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        fingerprint = _run_fingerprint(config, held_out_site, input_records)
        run_summary_path = model_dir / "run_summary.json"
        saved = (
            _load_completed_run(run_summary_path, fingerprint, dataset_dir, model_dir)
            if config.resume_completed_runs
            else None
        )
        if saved is not None:
            progress(
                f"[{run_index}/{len(held_out_sites)}] Reusing completed {run_name}."
            )
            result_rows = [
                {**row, "run_index": run_index, "run_name": run_name}
                for row in saved["results"]
            ]
            split_summary = saved["split_summary"]
        else:
            progress(
                f"[{run_index}/{len(held_out_sites)}] Creating 80/20 split with "
                f"'{held_out_site}' completely held out."
            )
            splits, split_summary = create_leave_one_site_split(
                source,
                held_out_site,
                config,
            )
            _save_split_datasets(splits, source_columns, dataset_dir, config)
            _write_json(dataset_dir / "split_summary.json", split_summary)

            progress(
                f"[{run_index}/{len(held_out_sites)}] Running one "
                f"{config.cv_splits}-fold search per model."
            )
            training = _train_site_models(
                splits,
                config,
                model_dir,
                held_out_site,
                progress,
            )
            result_rows = _result_rows(
                run_index,
                run_name,
                held_out_site,
                training,
                dataset_dir,
                model_dir,
            )
            pd.DataFrame(result_rows).to_csv(
                model_dir / "model_metrics.csv",
                index=False,
            )
            _write_json(
                run_summary_path,
                {
                    "fingerprint": fingerprint,
                    "run_index": run_index,
                    "run_name": run_name,
                    "held_out_site": held_out_site,
                    "split_summary": split_summary,
                    "results": result_rows,
                },
            )

        all_rows.extend(result_rows)
        run_records.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "held_out_site": held_out_site,
                "dataset_dir": str(dataset_dir),
                "model_dir": str(model_dir),
                "split_summary": split_summary,
            }
        )

    results = pd.DataFrame(all_rows).sort_values(
        ["run_index", "model"], kind="stable"
    )
    results_path = output_dir / "leave_one_site_out_results.csv"
    results.to_csv(results_path, index=False)

    summary = _summarize_results(results)
    summary_path = output_dir / "leave_one_site_out_summary_by_model.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "config": asdict(config),
        "available_sites": available_sites,
        "held_out_sites": held_out_sites,
        "input_files": input_records,
        "runs": run_records,
        "split_note": (
            "Train and test use disjoint scene IDs except for the configured overlap "
            "exception. Both are validated for site, sensor, season, and class coverage."
        ),
        "selection_note": (
            "Each classifier uses exactly one Optuna study and one fixed stratified "
            f"{config.cv_splits}-fold split per held-out site. Test and held-out-site "
            "data are used only after the best CV parameters are selected and refitted."
        ),
        "result_files": {
            "per_site_model_metrics": str(results_path),
            "summary_by_model": str(summary_path),
        },
    }
    manifest_path = output_dir / "leave_one_site_out_manifest.json"
    _write_json(manifest_path, manifest)
    progress(f"LOSO comparison written to {results_path}.")
    return {
        "results": results,
        "summary": summary,
        "manifest": manifest,
        "paths": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }


def create_leave_one_site_split(
    source: pd.DataFrame,
    held_out_site: str,
    config: LeaveOneSiteOutConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Create and validate a deterministic scene-aware 80/20 plus site holdout."""

    required = {config.site_column, config.scene_column, config.label_column}
    missing = sorted(required - set(source.columns))
    if missing:
        raise KeyError(f"Source data are missing required columns: {missing}")

    work = source.copy()
    if "_source_uid" not in work.columns:
        work["_source_uid"] = np.arange(len(work), dtype="int64")
    site_values = work[config.site_column].astype(str)
    leave_out = work.loc[site_values.eq(held_out_site)].copy()
    development = work.loc[~site_values.eq(held_out_site)].copy()
    if leave_out.empty:
        raise ValueError(f"Held-out site '{held_out_site}' has no rows.")
    if development.empty:
        raise ValueError("No development rows remain after holding out the site.")

    remaining, exception_train, exception_test, exception_summary = (
        _allocate_overlap_exception(
            development,
            held_out_site,
            config,
        )
    )
    test_scene_ids, scene_selection = _select_test_scenes(
        remaining,
        development,
        exception_test,
        config,
    )
    test_mask = remaining[config.scene_column].astype(str).isin(test_scene_ids)
    train = pd.concat(
        [remaining.loc[~test_mask], exception_train],
        ignore_index=True,
    )
    test = pd.concat(
        [remaining.loc[test_mask], exception_test],
        ignore_index=True,
    )
    splits = {
        "train": train.sort_values("_source_uid", kind="stable").reset_index(drop=True),
        "test": test.sort_values("_source_uid", kind="stable").reset_index(drop=True),
        "leave_out_site": leave_out.sort_values("_source_uid", kind="stable").reset_index(
            drop=True
        ),
    }
    validation = validate_leave_one_site_split(splits, held_out_site, config)
    summary = {
        "held_out_site": held_out_site,
        "total_rows": len(work),
        "development_rows": len(development),
        "requested_train_fraction_within_development": 1.0 - config.test_fraction,
        "requested_test_fraction_within_development": config.test_fraction,
        "actual_train_fraction_within_development": len(train) / len(development),
        "actual_test_fraction_within_development": len(test) / len(development),
        "scene_selection": scene_selection,
        "overlap_exception": exception_summary,
        "splits": {
            name: _split_summary(frame, config)
            for name, frame in splits.items()
        },
        "validation": validation,
    }
    return splits, _jsonable(summary)


def _allocate_overlap_exception(
    development: pd.DataFrame,
    held_out_site: str,
    config: LeaveOneSiteOutConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Remove and allocate the single permitted train/test scene overlap."""

    empty = development.iloc[0:0].copy()
    scene_id = config.overlap_exception_scene_id
    if scene_id is None:
        return development.copy(), empty, empty, {
            "configured": False,
            "applied": False,
        }

    if held_out_site == config.overlap_exception_site:
        return development.copy(), empty, empty, {
            "configured": True,
            "applied": False,
            "scene_id": scene_id,
            "site": config.overlap_exception_site,
            "class": config.overlap_exception_class,
            "reason": "The exception site is the held-out site.",
            "train_rows": 0,
            "test_rows": 0,
        }

    scene_mask = development[config.scene_column].astype(str).eq(scene_id)
    scene_rows = development.loc[scene_mask].sort_values(
        "_source_uid", kind="stable"
    )
    if scene_rows.empty:
        raise ValueError(
            f"Overlap-exception scene '{scene_id}' was not found in development data."
        )
    scene_sites = set(scene_rows[config.site_column].astype(str))
    if scene_sites != {config.overlap_exception_site}:
        raise ValueError(
            f"Overlap-exception scene '{scene_id}' must contain only site "
            f"'{config.overlap_exception_site}', found {sorted(scene_sites)}."
        )
    scene_classes = set(
        pd.to_numeric(scene_rows[config.label_column], errors="raise").astype(int)
    )
    if scene_classes != {config.overlap_exception_class}:
        raise ValueError(
            f"Overlap-exception scene '{scene_id}' must contain only class "
            f"{config.overlap_exception_class}, found {sorted(scene_classes)}."
        )

    train_rows = config.overlap_exception_train_rows
    test_rows = config.overlap_exception_test_rows
    expected_rows = train_rows + test_rows
    if len(scene_rows) != expected_rows:
        raise ValueError(
            f"Overlap-exception scene '{scene_id}' must have exactly {expected_rows} "
            f"rows, found {len(scene_rows)}."
        )
    exception_train = scene_rows.iloc[:train_rows].copy()
    exception_test = scene_rows.iloc[train_rows:].copy()
    remaining = development.loc[~scene_mask].copy()
    return remaining, exception_train, exception_test, {
        "configured": True,
        "applied": True,
        "scene_id": scene_id,
        "site": config.overlap_exception_site,
        "class": config.overlap_exception_class,
        "train_rows": len(exception_train),
        "test_rows": len(exception_test),
        "allocation_rule": (
            "Rows are ordered by source input and source row; the first configured "
            "rows go to training and the remainder go to test."
        ),
    }


def _select_test_scenes(
    remaining: pd.DataFrame,
    development: pd.DataFrame,
    exception_test: pd.DataFrame,
    config: LeaveOneSiteOutConfig,
) -> tuple[list[str], dict[str, Any]]:
    """Select whole test scenes while preserving required coverage in both splits."""

    scene_column = config.scene_column
    scene_ids = np.asarray(
        sorted(remaining[scene_column].astype(str).unique()),
        dtype=object,
    )
    total_scene_count = int(development[scene_column].astype(str).nunique())
    desired_test_scene_count = round(total_scene_count * config.test_fraction)
    exception_scene_count = int(not exception_test.empty)
    n_test_scenes = desired_test_scene_count - exception_scene_count
    if n_test_scenes <= 0 or n_test_scenes >= len(scene_ids):
        raise ValueError(
            "Requested test fraction is incompatible with scene-disjoint splitting."
        )

    full = _with_coverage_columns(development, config)
    available = _with_coverage_columns(remaining, config)
    exception = _with_coverage_columns(exception_test, config)
    specifications = [
        ("_coverage_site", sorted(full["_coverage_site"].unique()), 1.0),
        ("_coverage_sensor", sorted(full["_coverage_sensor"].unique()), 1.0),
        ("_coverage_season", list(SEASONS), 1.0),
        ("_coverage_class", sorted(full["_coverage_class"].unique()), 3.0),
    ]

    matrices: list[np.ndarray] = []
    totals: list[float] = []
    exception_counts: list[float] = []
    component_weights: list[float] = []
    category_labels: list[str] = []
    for column, categories, weight in specifications:
        missing_categories = sorted(set(categories) - set(full[column]))
        if missing_categories:
            raise ValueError(
                f"Development data are missing required {column} values: "
                f"{missing_categories}"
            )
        table = pd.crosstab(available[scene_column], available[column]).reindex(
            index=scene_ids,
            columns=categories,
            fill_value=0,
        )
        matrices.append(table.to_numpy(dtype="float64"))
        totals.extend(float(full[column].eq(category).sum()) for category in categories)
        exception_counts.extend(
            float(exception[column].eq(category).sum()) for category in categories
        )
        component_weights.extend([weight] * len(categories))
        category_labels.extend(f"{column}:{category}" for category in categories)

    balance_matrix = np.concatenate(matrices, axis=1)
    totals_array = np.asarray(totals, dtype="float64")
    exception_array = np.asarray(exception_counts, dtype="float64")
    weights = np.asarray(component_weights, dtype="float64")
    scene_row_counts = (
        remaining.groupby(scene_column).size().reindex(scene_ids).to_numpy(dtype="int64")
    )
    target_test_rows = round(len(development) * config.test_fraction)
    target_fraction = config.test_fraction
    rng = np.random.default_rng(config.split_random_state)
    best_score = np.inf
    best_iteration: int | None = None
    best_indices: np.ndarray | None = None
    best_test_rows: int | None = None

    for iteration in range(config.scene_split_search_iterations):
        indices = np.sort(
            rng.choice(len(scene_ids), size=n_test_scenes, replace=False)
        )
        selected = exception_array + balance_matrix[indices].sum(axis=0)
        remaining_counts = totals_array - selected
        if np.any(selected <= 0) or np.any(remaining_counts <= 0):
            continue
        selected_rows = int(len(exception_test) + scene_row_counts[indices].sum())
        row_fraction = selected_rows / len(development)
        if (
            abs(row_fraction - target_fraction)
            > config.scene_split_max_test_fraction_deviation
        ):
            continue
        fractions = selected / totals_array
        distribution_score = float(
            np.mean(weights * ((fractions - target_fraction) / target_fraction) ** 2)
        )
        score = distribution_score + 5.0 * (
            (row_fraction - target_fraction) / target_fraction
        ) ** 2
        if score < best_score:
            best_score = score
            best_iteration = iteration
            best_indices = indices.copy()
            best_test_rows = selected_rows

    if best_indices is None or best_test_rows is None:
        raise ValueError(
            "Could not find a scene-disjoint 80/20 split with complete site, sensor, "
            "season, and class coverage."
        )

    selected_scene_ids = sorted(str(scene_ids[index]) for index in best_indices)
    return selected_scene_ids, {
        "method": "deterministic randomized whole-scene search",
        "search_iterations": config.scene_split_search_iterations,
        "selected_iteration": best_iteration,
        "balance_score": best_score,
        "target_test_rows": target_test_rows,
        "actual_test_rows": best_test_rows,
        "target_test_fraction": target_fraction,
        "actual_test_fraction": best_test_rows / len(development),
        "maximum_test_fraction_deviation": (
            config.scene_split_max_test_fraction_deviation
        ),
        "target_test_scene_count": desired_test_scene_count,
        "whole_test_scene_count": len(selected_scene_ids),
        "overlap_exception_scene_count": exception_scene_count,
        "actual_test_scene_count": len(selected_scene_ids) + exception_scene_count,
        "test_scene_ids_excluding_exception": selected_scene_ids,
        "coverage_components": category_labels,
    }


def _with_coverage_columns(
    frame: pd.DataFrame,
    config: LeaveOneSiteOutConfig,
) -> pd.DataFrame:
    work = frame.copy()
    work["_coverage_site"] = work[config.site_column].astype(str)
    work["_coverage_sensor"] = _scene_sensor(work[config.scene_column])
    work["_coverage_season"] = _scene_season(work[config.scene_column])
    work["_coverage_class"] = (
        pd.to_numeric(work[config.label_column], errors="raise").astype(int).astype(str)
    )
    return work


def _scene_sensor(scene_ids: pd.Series) -> pd.Series:
    sensors = scene_ids.astype(str).str.split("_").str[0]
    invalid = ~sensors.isin(["LC08", "LC09"])
    if invalid.any():
        raise ValueError(
            f"Unsupported Landsat scene IDs: {scene_ids.loc[invalid].head(5).tolist()}"
        )
    return sensors


def _scene_season(scene_ids: pd.Series) -> pd.Series:
    dates = pd.to_datetime(
        scene_ids.astype(str).str.split("_").str[3],
        format="%Y%m%d",
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError(
            f"Could not parse acquisition dates from scene IDs: "
            f"{scene_ids.loc[dates.isna()].head(5).tolist()}"
        )
    months = dates.dt.month
    return pd.Series(
        np.select(
            [
                months.isin([12, 1, 2]),
                months.isin([3, 4, 5]),
                months.isin([6, 7, 8]),
            ],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        index=scene_ids.index,
        dtype="string",
    )


def validate_leave_one_site_split(
    splits: dict[str, pd.DataFrame],
    held_out_site: str,
    config: LeaveOneSiteOutConfig,
) -> dict[str, Any]:
    """Validate row accounting, site exclusivity, and train/test coverage."""

    expected_names = {"train", "test", "leave_out_site"}
    if set(splits) != expected_names:
        raise ValueError(f"Expected split names {sorted(expected_names)}.")

    combined = pd.concat(splits.values(), ignore_index=True)
    uids = [set(frame["_source_uid"].astype(int)) for frame in splits.values()]
    if any(uids[left] & uids[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("LOSO splits contain duplicated source rows.")
    if sum(len(values) for values in uids) != len(set().union(*uids)):
        raise ValueError("LOSO row accounting failed.")

    train_sites = set(splits["train"][config.site_column].astype(str))
    test_sites = set(splits["test"][config.site_column].astype(str))
    held_sites = set(splits["leave_out_site"][config.site_column].astype(str))
    development_sites = set(combined[config.site_column].astype(str)) - {held_out_site}
    if held_sites != {held_out_site}:
        raise ValueError("The leave-out-site dataset must contain exactly the held-out site.")
    if held_out_site in train_sites or held_out_site in test_sites:
        raise ValueError("Held-out-site rows leaked into training or test data.")
    if train_sites != development_sites or test_sites != development_sites:
        raise ValueError("Every remaining site must appear in both train and test data.")

    development = pd.concat([splits["train"], splits["test"]], ignore_index=True)
    expected_classes = set(CLASS_LABELS)
    expected_sensors = {"LC08", "LC09"}
    expected_seasons = set(SEASONS)
    coverage: dict[str, Any] = {}
    for name, frame in splits.items():
        classes = set(pd.to_numeric(frame[config.label_column]).astype(int))
        sensors = set(_scene_sensor(frame[config.scene_column]))
        seasons = set(_scene_season(frame[config.scene_column]))
        if name in {"train", "test"} and classes != expected_classes:
            raise ValueError(
                f"Split '{name}' does not contain all classes: {sorted(classes)}"
            )
        if name in {"train", "test"} and sensors != expected_sensors:
            raise ValueError(
                f"Split '{name}' does not contain all sensors: {sorted(sensors)}"
            )
        if name in {"train", "test"} and seasons != expected_seasons:
            raise ValueError(
                f"Split '{name}' does not contain all seasons: {sorted(seasons)}"
            )
        coverage[name] = {
            "sites": sorted(frame[config.site_column].astype(str).unique()),
            "classes": sorted(classes),
            "sensors": sorted(sensors),
            "seasons": sorted(seasons),
        }

    train_scenes = set(splits["train"][config.scene_column].astype(str))
    test_scenes = set(splits["test"][config.scene_column].astype(str))
    train_test_overlap = sorted(train_scenes & test_scenes)
    exception_scene_id = config.overlap_exception_scene_id
    exception_applies = (
        exception_scene_id is not None
        and held_out_site != config.overlap_exception_site
    )
    expected_train_test_overlap = {exception_scene_id} if exception_applies else set()
    if set(train_test_overlap) != expected_train_test_overlap:
        raise ValueError(
            "Train/test scene overlap must contain only the configured exception: "
            f"expected {sorted(expected_train_test_overlap)}, got {train_test_overlap}."
        )

    exception_counts = {
        name: int(frame[config.scene_column].astype(str).eq(exception_scene_id).sum())
        if exception_scene_id is not None
        else 0
        for name, frame in splits.items()
    }
    if exception_scene_id is not None:
        exception_rows = combined.loc[
            combined[config.scene_column].astype(str).eq(exception_scene_id)
        ]
        exception_sites = set(exception_rows[config.site_column].astype(str))
        exception_classes = set(
            pd.to_numeric(exception_rows[config.label_column], errors="raise").astype(int)
        )
        if exception_sites != {config.overlap_exception_site}:
            raise ValueError(
                "The overlap-exception scene does not belong exclusively to its "
                "configured site."
            )
        if exception_classes != {config.overlap_exception_class}:
            raise ValueError(
                "The overlap-exception scene does not contain exclusively its "
                "configured class."
            )
        if exception_applies:
            expected_exception_counts = {
                "train": config.overlap_exception_train_rows,
                "test": config.overlap_exception_test_rows,
                "leave_out_site": 0,
            }
        else:
            expected_exception_counts = {
                "train": 0,
                "test": 0,
                "leave_out_site": (
                    config.overlap_exception_train_rows
                    + config.overlap_exception_test_rows
                ),
            }
        if exception_counts != expected_exception_counts:
            raise ValueError(
                f"Overlap-exception allocation is incorrect: expected "
                f"{expected_exception_counts}, got {exception_counts}."
            )

    held_scenes = set(splits["leave_out_site"][config.scene_column].astype(str))
    development_scenes = train_scenes | test_scenes
    scene_overlap = sorted(held_scenes & development_scenes)

    return {
        "row_count": len(combined),
        "held_out_site_exclusive": True,
        "train_test_scene_overlap_count": len(train_test_overlap),
        "train_test_scene_overlap": train_test_overlap,
        "only_configured_train_test_overlap": True,
        "overlap_exception_counts": exception_counts,
        "shared_acquisition_scene_count": len(scene_overlap),
        "shared_acquisition_scenes": scene_overlap,
        "shared_acquisition_note": (
            "A scene ID may cover more than one lake. Shared IDs do not contain "
            "held-out-site pixels in development data; they are reported for audit."
        ),
        "coverage": coverage,
    }


def _train_site_models(
    splits: dict[str, pd.DataFrame],
    config: LeaveOneSiteOutConfig,
    model_dir: Path,
    held_out_site: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    data = _prepare_site_data(splits, config)
    cv = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.cv_random_state,
    )

    progress(f"  {held_out_site}: tuning Decision Tree ({config.n_trials_dt} trials).")
    dt_study, dt_model, dt_tuning = _tune_decision_tree(
        data.x_train, data.y_train, cv, config
    )
    progress(f"  {held_out_site}: tuning Random Forest ({config.n_trials_rf} trials).")
    rf_study, rf_model, rf_tuning = _tune_random_forest(
        data.x_train, data.y_train, cv, config
    )
    progress(f"  {held_out_site}: tuning XGBoost ({config.n_trials_xgb} trials).")
    xgb_study, xgb_model, xgb_tuning = _tune_xgboost(
        data.x_train, data.y_train, cv, config
    )

    models = {
        "decision_tree": dt_model,
        "random_forest": rf_model,
        "xgboost": xgb_model,
    }
    studies = {
        "decision_tree": dt_study,
        "random_forest": rf_study,
        "xgboost": xgb_study,
    }
    tuning = {
        "decision_tree": dt_tuning,
        "random_forest": rf_tuning,
        "xgboost": xgb_tuning,
    }
    metrics = {
        name: _evaluate_fitted_model(
            model,
            data,
            xgb_label_shift=name == "xgboost",
            tuning=tuning[name],
        )
        for name, model in models.items()
    }

    model_paths: dict[str, str] = {}
    for name, model in models.items():
        model_path = model_dir / MODEL_FILENAMES[name]
        joblib.dump(model, model_path)
        joblib.dump(studies[name], model_dir / f"{name}_optuna_study.pkl")
        studies[name].trials_dataframe().to_csv(
            model_dir / f"{name}_optuna_trials.csv",
            index=False,
        )
        _save_evaluation_tables(model_dir, name, metrics[name])
        model_paths[name] = str(model_path)

    metadata = {
        "config": asdict(config),
        "held_out_site": held_out_site,
        "feature_columns": data.feature_columns,
        "class_labels": CLASS_LABELS,
        "cv": {
            "kind": "StratifiedKFold over training pixels",
            "n_splits": config.cv_splits,
            "shuffle": True,
            "random_state": config.cv_random_state,
            "searches_per_model": 1,
            "selection_metric": config.scoring,
        },
        "test_used_for_selection": False,
        "leave_out_site_used_for_selection": False,
        "xgboost_label_shift": (
            "XGBoost is trained with labels 0,1,2 and evaluated as labels 1,2,3."
        ),
        "dropped_incomplete_rows": data.dropped_incomplete_rows,
        "tuning": tuning,
        "model_paths": model_paths,
    }
    _write_json(model_dir / "model_metadata.json", metadata)
    _write_json(model_dir / "metrics_summary.json", metrics)
    return {
        "models": models,
        "studies": studies,
        "metrics": metrics,
        "metadata": metadata,
    }


def _tune_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: LeaveOneSiteOutConfig,
) -> tuple[optuna.Study, DecisionTreeClassifier, dict[str, Any]]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2", 0.4, 0.6, 0.8]
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-9, 1e-2, log=True),
        }
        model = DecisionTreeClassifier(
            **params,
            random_state=config.model_random_state,
        )
        scores = _sklearn_cv_scores(model, x_train, y_train, cv, config.scoring)
        _record_trial_scores(trial, scores)
        return float(np.mean(scores))

    study = _single_study("DecisionTree_LOSO", config.optuna_seed)
    study.optimize(objective, n_trials=config.n_trials_dt, n_jobs=config.optuna_n_jobs)
    model = DecisionTreeClassifier(
        **study.best_params,
        random_state=config.model_random_state,
    )
    model.fit(x_train, y_train)
    return study, model, _tuning_summary(study, model.get_params(deep=False), config)


def _tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: LeaveOneSiteOutConfig,
) -> tuple[optuna.Study, RandomForestClassifier, dict[str, Any]]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [8, 12, 16, 24, 32, 48, None]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
        model = RandomForestClassifier(
            **params,
            random_state=config.model_random_state,
            n_jobs=config.rf_n_jobs,
        )
        scores = _sklearn_cv_scores(model, x_train, y_train, cv, config.scoring)
        _record_trial_scores(trial, scores)
        return float(np.mean(scores))

    study = _single_study("RandomForest_LOSO", config.optuna_seed + 1)
    study.optimize(objective, n_trials=config.n_trials_rf, n_jobs=config.optuna_n_jobs)
    model = RandomForestClassifier(
        **study.best_params,
        random_state=config.model_random_state,
        n_jobs=config.rf_n_jobs,
    )
    model.fit(x_train, y_train)
    return study, model, _tuning_summary(study, model.get_params(deep=False), config)


def _tune_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: LeaveOneSiteOutConfig,
) -> tuple[optuna.Study, xgb.XGBClassifier, dict[str, Any]]:
    y_zero = y_train - 1

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.015, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.5, 30.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        }
        scores, best_iterations = _xgb_cv_scores(
            params,
            x_train,
            y_zero,
            cv,
            config,
        )
        _record_trial_scores(trial, scores)
        trial.set_user_attr("fold_best_iterations", best_iterations)
        return float(np.mean(scores))

    study = _single_study("XGBoost_LOSO", config.optuna_seed + 2)
    study.optimize(objective, n_trials=config.n_trials_xgb, n_jobs=config.optuna_n_jobs)
    fold_iterations = study.best_trial.user_attrs["fold_best_iterations"]
    final_n_estimators = int(
        np.clip(
            round(float(np.median(fold_iterations))),
            1,
            config.xgb_max_estimators,
        )
    )
    final_params = {**study.best_params, "n_estimators": final_n_estimators}
    model = _new_xgb_model(final_params, config, early_stopping=False)
    model.fit(
        x_train,
        y_zero,
        sample_weight=_sample_weight(y_zero, config.xgb_sample_weight),
    )
    summary = _tuning_summary(study, final_params, config)
    summary.update(
        {
            "early_stopping_rounds": config.xgb_early_stopping_rounds,
            "maximum_estimators_during_cv": config.xgb_max_estimators,
            "best_trial_fold_iterations": fold_iterations,
            "final_n_estimators": final_n_estimators,
        }
    )
    return study, model, summary


def _single_study(
    name: str,
    seed: int,
) -> optuna.Study:
    return optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )


def _sklearn_cv_scores(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    scoring: str,
) -> list[float]:
    scores: list[float] = []
    for train_index, valid_index in cv.split(x, y):
        fitted = clone(model)
        fitted.fit(x.iloc[train_index], y.iloc[train_index])
        pred = fitted.predict(x.iloc[valid_index])
        scores.append(_score_predictions(y.iloc[valid_index], pred, scoring))
    return scores


def _xgb_cv_scores(
    params: dict[str, Any],
    x: pd.DataFrame,
    y_zero: pd.Series,
    cv: StratifiedKFold,
    config: LeaveOneSiteOutConfig,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    best_iterations: list[int] = []
    for train_index, valid_index in cv.split(x, y_zero):
        model = _new_xgb_model(params, config, early_stopping=True)
        y_fold = y_zero.iloc[train_index]
        y_valid = y_zero.iloc[valid_index]
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(x.iloc[valid_index], y_valid)],
            "verbose": False,
        }
        train_weight = _sample_weight(y_fold, config.xgb_sample_weight)
        if train_weight is not None:
            fit_kwargs["sample_weight"] = train_weight
            fit_kwargs["sample_weight_eval_set"] = [
                _sample_weight(y_valid, config.xgb_sample_weight)
            ]
        model.fit(x.iloc[train_index], y_fold, **fit_kwargs)
        pred = model.predict(x.iloc[valid_index])
        scores.append(_score_predictions(y_valid, pred, config.scoring))
        best_iteration = getattr(model, "best_iteration", None)
        best_iterations.append(
            config.xgb_max_estimators
            if best_iteration is None
            else int(best_iteration) + 1
        )
    return scores, best_iterations


def _new_xgb_model(
    params: dict[str, Any],
    config: LeaveOneSiteOutConfig,
    *,
    early_stopping: bool,
) -> xgb.XGBClassifier:
    model_params = {
        **params,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": config.model_random_state,
        "n_jobs": config.xgb_n_jobs,
    }
    if early_stopping:
        model_params["n_estimators"] = config.xgb_max_estimators
        model_params["early_stopping_rounds"] = config.xgb_early_stopping_rounds
    return xgb.XGBClassifier(**model_params)


def _record_trial_scores(trial: optuna.Trial, scores: list[float]) -> None:
    values = np.asarray(scores, dtype="float64")
    trial.set_user_attr("fold_scores", [float(value) for value in values])
    trial.set_user_attr("cv_mean", float(values.mean()))
    trial.set_user_attr("cv_std", float(values.std()))


def _tuning_summary(
    study: optuna.Study,
    final_parameters: dict[str, Any],
    config: LeaveOneSiteOutConfig,
) -> dict[str, Any]:
    return {
        "searches": 1,
        "n_trials": len(study.trials),
        "cv_splits": config.cv_splits,
        "scoring": config.scoring,
        "best_trial_number": study.best_trial.number,
        "best_cv_mean": study.best_trial.user_attrs["cv_mean"],
        "best_cv_std": study.best_trial.user_attrs["cv_std"],
        "best_cv_fold_scores": study.best_trial.user_attrs["fold_scores"],
        "parameters": final_parameters,
    }


def _prepare_site_data(
    splits: dict[str, pd.DataFrame],
    config: LeaveOneSiteOutConfig,
) -> PreparedSiteData:
    drop_columns = set(config.drop_columns) | {
        config.site_column,
        config.scene_column,
        "_source_uid",
        "_source_csv",
        "_source_row",
    }

    def prepare(
        frame: pd.DataFrame,
        name: str,
        feature_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, int]:
        y = pd.to_numeric(frame[config.label_column], errors="coerce")
        x = frame.drop(columns=[config.label_column]).drop(
            columns=list(drop_columns), errors="ignore"
        )
        x = x.apply(pd.to_numeric, errors="coerce")
        if feature_columns is not None:
            missing = sorted(set(feature_columns) - set(x.columns))
            if missing:
                raise ValueError(f"{name} data are missing model features: {missing}")
            x = x[feature_columns]
        keep = y.notna() & x.notna().all(axis=1)
        x = x.loc[keep].copy()
        y = y.loc[keep].astype(int).copy()
        if x.empty:
            raise ValueError(f"{name} data contain no complete model rows.")
        unexpected = sorted(set(y.unique()) - set(CLASS_LABELS))
        if unexpected:
            raise ValueError(f"{name} data contain unexpected labels: {unexpected}")
        return x, y, int((~keep).sum())

    x_train, y_train, train_dropped = prepare(splits["train"], "train")
    feature_columns = list(x_train.columns)
    x_test, y_test, test_dropped = prepare(
        splits["test"], "test", feature_columns
    )
    x_leave, y_leave, leave_dropped = prepare(
        splits["leave_out_site"], "leave_out_site", feature_columns
    )
    if y_train.value_counts().min() < config.cv_splits:
        raise ValueError("Every training class needs at least cv_splits rows.")
    return PreparedSiteData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_leave_out_site=x_leave,
        y_leave_out_site=y_leave,
        feature_columns=feature_columns,
        dropped_incomplete_rows={
            "train": train_dropped,
            "test": test_dropped,
            "leave_out_site": leave_dropped,
        },
    )


def _evaluate_fitted_model(
    model: Any,
    data: PreparedSiteData,
    *,
    xgb_label_shift: bool,
    tuning: dict[str, Any],
) -> dict[str, Any]:
    return {
        "datasets": {
            "train": _evaluate_dataset(
                model, data.x_train, data.y_train, xgb_label_shift
            ),
            "test": _evaluate_dataset(
                model, data.x_test, data.y_test, xgb_label_shift
            ),
            "leave_out_site": _evaluate_dataset(
                model,
                data.x_leave_out_site,
                data.y_leave_out_site,
                xgb_label_shift,
            ),
        },
        "tuning_cv": tuning,
    }


def _evaluate_dataset(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    xgb_label_shift: bool,
) -> dict[str, Any]:
    pred_model = model.predict(x)
    pred = np.asarray(pred_model, dtype="int64") + (1 if xgb_label_shift else 0)
    labels = [1, 2, 3]
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "n_rows": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y,
            pred,
            labels=labels,
            target_names=[CLASS_LABELS[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm,
        "producer_user_accuracy": _producer_user_accuracy(cm, labels),
        "binary_cloud_accuracy": _binary_cloud_metrics(y.to_numpy(), pred),
    }


def _save_evaluation_tables(
    output_dir: Path,
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    for dataset_name, dataset in metrics["datasets"].items():
        pd.DataFrame(
            dataset["confusion_matrix"],
            index=[f"true_{label}" for label in (1, 2, 3)],
            columns=[f"pred_{label}" for label in (1, 2, 3)],
        ).to_csv(output_dir / f"{model_name}_{dataset_name}_confusion_matrix.csv")
        pd.DataFrame(dataset["classification_report"]).transpose().to_csv(
            output_dir / f"{model_name}_{dataset_name}_classification_report.csv"
        )


def _result_rows(
    run_index: int,
    run_name: str,
    held_out_site: str,
    training: dict[str, Any],
    dataset_dir: Path,
    model_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in training["metrics"].items():
        datasets = metrics["datasets"]
        tuning = metrics["tuning_cv"]
        rows.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "held_out_site": held_out_site,
                "model": model_name,
                "train_accuracy": datasets["train"]["accuracy"],
                "train_balanced_accuracy": datasets["train"]["balanced_accuracy"],
                "train_f1_macro": datasets["train"]["f1_macro"],
                "test_accuracy": datasets["test"]["accuracy"],
                "test_balanced_accuracy": datasets["test"]["balanced_accuracy"],
                "test_f1_macro": datasets["test"]["f1_macro"],
                "leave_out_site_accuracy": datasets["leave_out_site"]["accuracy"],
                "leave_out_site_balanced_accuracy": datasets["leave_out_site"][
                    "balanced_accuracy"
                ],
                "leave_out_site_f1_macro": datasets["leave_out_site"]["f1_macro"],
                "cv_selection_score_mean": tuning["best_cv_mean"],
                "cv_selection_score_std": tuning["best_cv_std"],
                "selection_scoring": tuning["scoring"],
                "n_trials": tuning["n_trials"],
                "searches": tuning["searches"],
                "final_n_estimators": tuning.get("final_n_estimators"),
                "selected_parameters": json.dumps(
                    tuning["parameters"], sort_keys=True, default=str
                ),
                "train_csv": str(dataset_dir / DATASET_FILENAMES["train"]),
                "test_csv": str(dataset_dir / DATASET_FILENAMES["test"]),
                "leave_out_site_csv": str(
                    dataset_dir / DATASET_FILENAMES["leave_out_site"]
                ),
                "scene_manifest_path": str(
                    dataset_dir / "scene_split_manifest.csv"
                ),
                "model_path": str(model_dir / MODEL_FILENAMES[model_name]),
                "metrics_path": str(model_dir / "metrics_summary.json"),
                "metadata_path": str(model_dir / "model_metadata.json"),
            }
        )
    return rows


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "leave_out_site_balanced_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in results.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model_name, "n_sites": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _save_split_datasets(
    splits: dict[str, pd.DataFrame],
    source_columns: list[str],
    dataset_dir: Path,
    config: LeaveOneSiteOutConfig,
) -> None:
    for name, frame in splits.items():
        frame[source_columns].to_csv(
            dataset_dir / DATASET_FILENAMES[name],
            index=False,
        )
    _scene_partition_manifest(splits, config).to_csv(
        dataset_dir / "scene_split_manifest.csv",
        index=False,
    )


def _scene_partition_manifest(
    splits: dict[str, pd.DataFrame],
    config: LeaveOneSiteOutConfig,
) -> pd.DataFrame:
    frames = [frame.assign(_partition=name) for name, frame in splits.items()]
    combined = _with_coverage_columns(pd.concat(frames, ignore_index=True), config)
    rows: list[dict[str, Any]] = []
    for scene_id, group in combined.groupby(config.scene_column, sort=True):
        partition_counts = group["_partition"].value_counts()
        present = [
            name for name in DATASET_FILENAMES if int(partition_counts.get(name, 0)) > 0
        ]
        rows.append(
            {
                "scene_id": scene_id,
                "sensor": group["_coverage_sensor"].iloc[0],
                "season": group["_coverage_season"].iloc[0],
                "sites": ";".join(sorted(group["_coverage_site"].unique())),
                "classes": ";".join(sorted(group["_coverage_class"].unique())),
                "partitions": ";".join(present),
                "train_rows": int(partition_counts.get("train", 0)),
                "test_rows": int(partition_counts.get("test", 0)),
                "leave_out_site_rows": int(
                    partition_counts.get("leave_out_site", 0)
                ),
                "is_train_test_overlap": "train" in present and "test" in present,
                "is_configured_overlap_exception": (
                    scene_id == config.overlap_exception_scene_id
                ),
            }
        )
    return pd.DataFrame(rows)


def _split_summary(frame: pd.DataFrame, config: LeaveOneSiteOutConfig) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "sites": int(frame[config.site_column].nunique()),
        "site_counts": frame[config.site_column].astype(str).value_counts().sort_index().to_dict(),
        "scenes": int(frame[config.scene_column].nunique()),
        "class_counts": (
            pd.to_numeric(frame[config.label_column]).astype(int).value_counts().sort_index().to_dict()
        ),
        "sensor_counts": (
            frame[config.scene_column].astype(str).str[:4].value_counts().sort_index().to_dict()
        ),
        "season_counts": (
            _scene_season(frame[config.scene_column])
            .value_counts()
            .reindex(SEASONS, fill_value=0)
            .to_dict()
        ),
    }


def _load_source_data(
    config: LeaveOneSiteOutConfig,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    reference_columns: list[str] | None = None
    offset = 0
    for csv_path in config.input_csvs:
        path = Path(csv_path)
        frame = pd.read_csv(path)
        if reference_columns is None:
            reference_columns = list(frame.columns)
        elif list(frame.columns) != reference_columns:
            raise ValueError(f"Input CSV columns differ: {path}")
        required = {config.site_column, config.scene_column, config.label_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Input CSV '{path}' is missing columns: {missing}")
        part = frame.copy()
        part["_source_csv"] = str(path)
        part["_source_row"] = np.arange(len(part), dtype="int64")
        part["_source_uid"] = np.arange(offset, offset + len(part), dtype="int64")
        offset += len(part)
        frames.append(part)
        input_records.append(
            {"path": str(path), "rows": len(frame), "sha256": _sha256(path)}
        )
    if not frames or reference_columns is None:
        raise ValueError("At least one input CSV is required.")
    source = pd.concat(frames, ignore_index=True)
    if source["_source_uid"].duplicated().any():
        raise RuntimeError("Internal source row identifiers are not unique.")
    return source, reference_columns, input_records


def _resolved_config(
    config: LeaveOneSiteOutConfig,
    root: Path,
) -> LeaveOneSiteOutConfig:
    return replace(
        config,
        input_csvs=[str(_resolve_path(root, path)) for path in config.input_csvs],
        output_dir=str(_resolve_path(root, config.output_dir)),
    )


def _validate_config(config: LeaveOneSiteOutConfig) -> None:
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between zero and one.")
    if config.cv_splits < 2:
        raise ValueError("cv_splits must be at least two.")
    if config.scene_split_search_iterations <= 0:
        raise ValueError("scene_split_search_iterations must be positive.")
    if not 0.0 < config.scene_split_max_test_fraction_deviation < 1.0:
        raise ValueError(
            "scene_split_max_test_fraction_deviation must be between zero and one."
        )
    exception_values = (
        config.overlap_exception_site,
        config.overlap_exception_class,
        config.overlap_exception_train_rows,
        config.overlap_exception_test_rows,
    )
    if config.overlap_exception_scene_id is None:
        if exception_values != (None, None, 0, 0):
            raise ValueError(
                "Overlap-exception site and row counts require an exception scene ID."
            )
    elif (
        config.overlap_exception_site is None
        or config.overlap_exception_class is None
        or config.overlap_exception_train_rows <= 0
        or config.overlap_exception_test_rows <= 0
    ):
        raise ValueError(
            "A configured overlap exception requires a site and positive train/test rows."
        )
    if config.scoring not in {"accuracy", "balanced_accuracy", "f1_macro"}:
        raise ValueError("scoring must be accuracy, balanced_accuracy, or f1_macro.")
    if min(config.n_trials_dt, config.n_trials_rf, config.n_trials_xgb) <= 0:
        raise ValueError("Every classifier must have at least one Optuna trial.")
    if config.optuna_n_jobs <= 0:
        raise ValueError("optuna_n_jobs must be positive.")
    if config.xgb_max_estimators <= 0 or config.xgb_early_stopping_rounds <= 0:
        raise ValueError("XGBoost estimator and early-stopping values must be positive.")
    if config.xgb_sample_weight not in {None, "balanced"}:
        raise ValueError("xgb_sample_weight must be null or 'balanced'.")
    if not config.input_csvs:
        raise ValueError("At least one input CSV must be configured.")
    if len(config.input_csvs) != len(set(config.input_csvs)):
        raise ValueError("input_csvs must not contain duplicate paths.")
    if config.held_out_sites is not None and not config.held_out_sites:
        raise ValueError("held_out_sites must be null or contain at least one site.")
    if config.held_out_sites is not None and len(config.held_out_sites) != len(
        set(config.held_out_sites)
    ):
        raise ValueError("held_out_sites must not contain duplicates.")


def _load_completed_run(
    summary_path: Path,
    fingerprint: str,
    dataset_dir: Path,
    model_dir: Path,
) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing LOSO run has a different configuration: {model_dir}. "
            "Use a new output_dir or disable resume_completed_runs."
        )
    required = [
        *(dataset_dir / filename for filename in DATASET_FILENAMES.values()),
        dataset_dir / "split_summary.json",
        dataset_dir / "scene_split_manifest.csv",
        *(model_dir / filename for filename in MODEL_FILENAMES.values()),
        model_dir / "metrics_summary.json",
        model_dir / "model_metadata.json",
        model_dir / "model_metrics.csv",
    ]
    for model_name in MODEL_FILENAMES:
        required.extend(
            [
                model_dir / f"{model_name}_optuna_study.pkl",
                model_dir / f"{model_name}_optuna_trials.csv",
                *(
                    model_dir / f"{model_name}_{dataset_name}_{report_name}.csv"
                    for dataset_name in DATASET_FILENAMES
                    for report_name in (
                        "classification_report",
                        "confusion_matrix",
                    )
                ),
            ]
        )
    return saved if all(path.exists() for path in required) else None


def _run_fingerprint(
    config: LeaveOneSiteOutConfig,
    held_out_site: str,
    input_records: list[dict[str, Any]],
) -> str:
    run_config = asdict(config)
    # These control orchestration, not the scientific contents of one site run.
    run_config.pop("held_out_sites", None)
    run_config.pop("resume_completed_runs", None)
    payload = {
        "config": run_config,
        "held_out_site": held_out_site,
        "input_files": input_records,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_predictions(y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if scoring == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unsupported scoring: {scoring}")


def _sample_weight(y: pd.Series, mode: str | None) -> np.ndarray | None:
    if mode is None:
        return None
    if mode != "balanced":
        raise ValueError(f"Unsupported sample-weight mode: {mode}")
    return compute_sample_weight(class_weight="balanced", y=y)


def _producer_user_accuracy(
    cm: np.ndarray,
    labels: list[int],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        result[str(label)] = {
            "label": CLASS_LABELS[label],
            "producer_accuracy": tp / (tp + fn) if tp + fn else None,
            "user_accuracy": tp / (tp + fp) if tp + fp else None,
        }
    return result


def _binary_cloud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    true_cloud = np.isin(y_true, [1, 2]).astype("uint8")
    pred_cloud = np.isin(y_pred, [1, 2]).astype("uint8")
    cm = confusion_matrix(true_cloud, pred_cloud, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": (tp + tn) / cm.sum() if cm.sum() else None,
        "producer_accuracy_cloud": tp / (tp + fn) if tp + fn else None,
        "user_accuracy_cloud": tp / (tp + fp) if tp + fp else None,
    }


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower())


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value
