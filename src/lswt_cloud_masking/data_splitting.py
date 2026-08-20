"""Reproducible scene-aware 80/10/10 splitting for model development."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEASONS = ("DJF", "MAM", "JJA", "SON")


@dataclass
class SplitConfig:
    """Configuration for the reproducible 80/10/10 data partition."""

    input_csvs: list[str]
    train_csv: str
    test_csv: str
    los_test_csv: str
    manifest_csv: str
    summary_json: str
    scene_column: str = "scene_id"
    lake_column: str = "lakeN"
    label_column: str = "lst_class"
    train_fraction: float = 0.8
    test_fraction: float = 0.1
    los_fraction: float = 0.1
    random_state: int = 42
    los_search_iterations: int = 100_000
    preserve_unique_lake_class_scenes: bool = True
    protected_train_scene_ids: list[str] = field(default_factory=list)
    protected_scene_familiar_test_rows: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


def run_split_pipeline(config: SplitConfig) -> dict[str, Any]:
    """Create, validate, save, and describe an 80/10/10 partition."""

    _validate_fractions(config)
    input_paths = [Path(path) for path in config.input_csvs]
    frames = [pd.read_csv(path) for path in input_paths]
    if not frames:
        raise ValueError("At least one input CSV is required.")
    _validate_input_columns(frames, config)

    source_columns = list(frames[0].columns)
    combined_parts: list[pd.DataFrame] = []
    source_offset = 0
    for path, frame in zip(input_paths, frames, strict=True):
        part = frame.copy()
        part["_source_csv"] = str(path)
        part["_source_row"] = np.arange(len(part), dtype="int64")
        part["_source_order"] = np.arange(source_offset, source_offset + len(part), dtype="int64")
        source_offset += len(part)
        combined_parts.append(part)
    combined = pd.concat(combined_parts, ignore_index=True)
    metadata = _scene_metadata(combined, config)

    protected = set(config.protected_train_scene_ids)
    available_scenes = set(metadata[config.scene_column])
    missing_protected = sorted(protected - available_scenes)
    if missing_protected:
        raise ValueError(f"Protected training scenes were not found: {missing_protected}")

    los_scene_ids, los_search = _select_los_scenes(combined, metadata, config)
    los_mask = combined[config.scene_column].astype(str).isin(los_scene_ids)
    los_test = combined.loc[los_mask].copy()
    development = combined.loc[~los_mask].copy()

    protected_mask = development[config.scene_column].astype(str).isin(protected)
    protected_rows = development.loc[protected_mask].copy()
    protected_train_parts: list[pd.DataFrame] = []
    protected_test_parts: list[pd.DataFrame] = []
    for scene_id in sorted(protected):
        scene_rows = protected_rows.loc[
            protected_rows[config.scene_column].astype(str).eq(scene_id)
        ].sort_values("_source_order", kind="stable")
        familiar_rows = config.protected_scene_familiar_test_rows.get(scene_id, 0)
        if familiar_rows >= len(scene_rows):
            raise ValueError(
                f"Protected scene '{scene_id}' must retain at least one training row."
            )
        if familiar_rows:
            protected_test_parts.append(scene_rows.tail(familiar_rows))
            protected_train_parts.append(scene_rows.iloc[:-familiar_rows])
        else:
            protected_train_parts.append(scene_rows)
    protected_train = (
        pd.concat(protected_train_parts, ignore_index=True)
        if protected_train_parts
        else protected_rows.iloc[0:0].copy()
    )
    protected_test = (
        pd.concat(protected_test_parts, ignore_index=True)
        if protected_test_parts
        else protected_rows.iloc[0:0].copy()
    )
    splittable = development.loc[~protected_mask].copy()
    target_test_rows = round(len(combined) * config.test_fraction) - len(protected_test)
    if target_test_rows <= 0 or target_test_rows >= len(splittable):
        raise ValueError("Requested familiar-test size is incompatible with the available rows.")

    strata = (
        splittable[config.scene_column].astype(str)
        + "|"
        + splittable[config.label_column].astype(int).astype(str)
    )
    train_part, familiar_test_part = train_test_split(
        splittable,
        test_size=target_test_rows,
        random_state=config.random_state,
        stratify=strata,
    )
    train = pd.concat([train_part, protected_train], ignore_index=True)
    familiar_test = pd.concat(
        [familiar_test_part, protected_test],
        ignore_index=True,
    )

    split_frames = {
        "train": _restore_output_order(train, source_columns),
        "test": _restore_output_order(familiar_test, source_columns),
        "los_test": _restore_output_order(los_test, source_columns),
    }
    validation = validate_splits(split_frames, config)
    manifest = _build_manifest(combined, split_frames, metadata, protected, config)

    output_paths = {
        "train": Path(config.train_csv),
        "test": Path(config.test_csv),
        "los_test": Path(config.los_test_csv),
        "manifest": Path(config.manifest_csv),
        "summary": Path(config.summary_json),
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "los_test"):
        split_frames[name].to_csv(output_paths[name], index=False)
    manifest.to_csv(output_paths["manifest"], index=False)

    summary = {
        "config": _portable_config(config),
        "input_files": [
            {"path": _portable_path(path), "sha256": _sha256(path), "rows": len(frame)}
            for path, frame in zip(input_paths, frames, strict=True)
        ],
        "total_rows": len(combined),
        "total_scenes": int(combined[config.scene_column].nunique()),
        "splits": {
            name: _split_summary(frame, len(combined), config)
            for name, frame in split_frames.items()
        },
        "scene_overlap": validation["scene_overlap"],
        "coverage": validation["coverage"],
        "protected_train_scene_ids": sorted(protected),
        "los_selection": los_search,
        "output_files": {
            name: {"path": _portable_path(path), "sha256": _sha256(path)}
            for name, path in output_paths.items()
            if name != "summary"
        },
    }
    summary = _jsonable(summary)
    output_paths["summary"].write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return {
        "splits": split_frames,
        "manifest": manifest,
        "summary": summary,
        "paths": {name: str(path) for name, path in output_paths.items()},
    }


def validate_splits(
    split_frames: dict[str, pd.DataFrame],
    config: SplitConfig,
) -> dict[str, Any]:
    """Validate coverage, exclusivity, protected scenes, and row accounting."""

    required_names = {"train", "test", "los_test"}
    if set(split_frames) != required_names:
        raise ValueError(f"Expected split names {sorted(required_names)}, got {sorted(split_frames)}")

    scene_sets = {
        name: set(frame[config.scene_column].astype(str))
        for name, frame in split_frames.items()
    }
    overlaps = {
        "train_test": sorted(scene_sets["train"] & scene_sets["test"]),
        "train_los_test": sorted(scene_sets["train"] & scene_sets["los_test"]),
        "test_los_test": sorted(scene_sets["test"] & scene_sets["los_test"]),
    }
    if overlaps["train_los_test"] or overlaps["test_los_test"]:
        raise ValueError("LOS scenes must be absent from both training and familiar-test data.")

    protected = set(config.protected_train_scene_ids)
    if not protected.issubset(scene_sets["train"]):
        raise ValueError("Every protected scene must appear in training.")
    if protected & scene_sets["los_test"]:
        raise ValueError("Protected training scenes cannot appear in the LOS test split.")
    for scene_id in protected:
        expected_test_rows = config.protected_scene_familiar_test_rows.get(scene_id, 0)
        actual_test_rows = int(
            split_frames["test"][config.scene_column].astype(str).eq(scene_id).sum()
        )
        if actual_test_rows != expected_test_rows:
            raise ValueError(
                f"Protected scene '{scene_id}' should have {expected_test_rows} familiar-test "
                f"rows, found {actual_test_rows}."
            )

    reference = pd.concat(split_frames.values(), ignore_index=True)
    total_rows = len(reference)
    expected_sizes = {
        "test": round(total_rows * config.test_fraction),
        "los_test": round(total_rows * config.los_fraction),
    }
    expected_sizes["train"] = total_rows - expected_sizes["test"] - expected_sizes["los_test"]
    actual_sizes = {name: len(frame) for name, frame in split_frames.items()}
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Split row counts do not match the requested fractions: "
            f"expected {expected_sizes}, got {actual_sizes}."
        )

    expected = {
        "lakes": sorted(reference[config.lake_column].astype(str).unique()),
        "classes": sorted(reference[config.label_column].astype(int).unique()),
        "sensors": sorted(_sensor(reference[config.scene_column]).unique()),
        "seasons": list(SEASONS),
    }
    coverage: dict[str, Any] = {}
    for name, frame in split_frames.items():
        actual = {
            "lakes": sorted(frame[config.lake_column].astype(str).unique()),
            "classes": sorted(frame[config.label_column].astype(int).unique()),
            "sensors": sorted(_sensor(frame[config.scene_column]).unique()),
            "seasons": sorted(_season(_acquisition_date(frame[config.scene_column])).unique()),
        }
        missing = {
            key: sorted(set(expected[key]) - set(values))
            for key, values in actual.items()
            if set(expected[key]) - set(values)
        }
        if missing:
            raise ValueError(f"Split '{name}' is missing required coverage: {missing}")
        coverage[name] = actual

    return {
        "scene_overlap": {
            key: {"n_overlap": len(values), "examples": values[:10]}
            for key, values in overlaps.items()
        },
        "coverage": coverage,
    }


def _select_los_scenes(
    combined: pd.DataFrame,
    metadata: pd.DataFrame,
    config: SplitConfig,
) -> tuple[list[str], dict[str, Any]]:
    scene_col = config.scene_column
    lake_col = config.lake_column
    label_col = config.label_column
    protected = set(config.protected_train_scene_ids)
    work = combined.copy()
    work["_sensor"] = _sensor(work[scene_col])
    work["_season"] = _season(_acquisition_date(work[scene_col]))
    classes = sorted(work[label_col].astype(int).unique())
    lakes = sorted(work[lake_col].astype(str).unique())
    sensors = sorted(work["_sensor"].unique())
    seasons = list(SEASONS)

    development_coverage_scenes: set[str] = set()
    if config.preserve_unique_lake_class_scenes:
        scene_presence = work[[scene_col, lake_col, label_col]].drop_duplicates()
        pair_scene_counts = scene_presence.groupby([lake_col, label_col])[scene_col].transform(
            "nunique"
        )
        development_coverage_scenes = set(
            scene_presence.loc[pair_scene_counts.eq(1), scene_col].astype(str)
        )

    excluded = protected | development_coverage_scenes
    scene_ids = np.asarray(sorted(set(metadata[scene_col]) - excluded), dtype=object)
    n_los_scenes = round(metadata[scene_col].nunique() * config.los_fraction)
    if n_los_scenes <= 0 or n_los_scenes >= len(scene_ids):
        raise ValueError("Requested LOS scene count is incompatible with the available scenes.")

    block_specs = [
        (label_col, classes, 3.0),
        (lake_col, lakes, 1.0),
        ("_sensor", sensors, 1.0),
        ("_season", seasons, 1.0),
    ]
    blocks: list[np.ndarray] = []
    component_weights: list[float] = []
    for column, categories, weight in block_specs:
        table = pd.crosstab(work[scene_col], work[column]).reindex(
            index=scene_ids,
            columns=categories,
            fill_value=0,
        )
        blocks.append(table.to_numpy(dtype="float64"))
        component_weights.extend([weight] * len(categories))
    balance_matrix = np.concatenate(blocks, axis=1)
    totals = balance_matrix.sum(axis=0)
    weights = np.asarray(component_weights, dtype="float64")

    lake_class_categories = pd.MultiIndex.from_product([lakes, classes])
    lake_class = pd.crosstab(
        work[scene_col],
        [work[lake_col].astype(str), work[label_col].astype(int)],
    ).reindex(index=scene_ids, columns=lake_class_categories, fill_value=0)
    lake_class_matrix = lake_class.to_numpy(dtype="float64")
    lake_class_totals = lake_class_matrix.sum(axis=0)
    lake_class_scene_counts = (lake_class_matrix > 0).sum(axis=0)
    feasible_los_lake_classes = lake_class_scene_counts >= 2

    rng = np.random.default_rng(config.random_state)
    best_score = np.inf
    best_iteration: int | None = None
    best_indices: np.ndarray | None = None
    label_count = len(classes)
    coverage_start = label_count
    target = config.los_fraction
    target_los_rows = round(len(work) * target)
    for iteration in range(config.los_search_iterations):
        indices = np.sort(rng.choice(len(scene_ids), size=n_los_scenes, replace=False))
        selected = balance_matrix[indices].sum(axis=0)
        selected_rows = int(selected[:label_count].sum())
        if selected_rows != target_los_rows:
            continue
        if np.any(selected[coverage_start:] == 0):
            continue
        missing_lake_classes = int(
            np.sum(
                (lake_class_matrix[indices].sum(axis=0) == 0)
                & feasible_los_lake_classes
            )
        )
        missing_development_lake_classes = int(
            np.sum(
                (lake_class_totals - lake_class_matrix[indices].sum(axis=0) == 0)
                & (lake_class_totals > 0)
            )
        )
        fractions = selected / totals
        score = float(np.mean(weights * ((fractions - target) / target) ** 2))
        row_fraction = float(selected_rows / len(work))
        score += 5.0 * ((row_fraction - target) / target) ** 2
        score += 2.0 * missing_lake_classes
        score += 20.0 * missing_development_lake_classes
        if score < best_score:
            best_score = score
            best_iteration = iteration
            best_indices = indices.copy()

    if best_indices is None:
        raise ValueError(
            "Could not find a LOS scene set with complete lake, sensor, season, and class coverage."
        )
    selected_ids = sorted(str(scene_ids[index]) for index in best_indices)
    return selected_ids, {
        "search_iterations": config.los_search_iterations,
        "selected_iteration": best_iteration,
        "balance_score": best_score,
        "n_los_scenes": len(selected_ids),
        "target_los_rows": target_los_rows,
        "scene_ids": selected_ids,
        "development_coverage_scene_ids": sorted(development_coverage_scenes),
    }


def _scene_metadata(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    scene_col = config.scene_column
    work = frame.copy()
    work[scene_col] = work[scene_col].astype(str)
    work["sensor"] = _sensor(work[scene_col])
    work["acquisition_date"] = _acquisition_date(work[scene_col])
    work["season"] = _season(work["acquisition_date"])
    return (
        work.groupby(scene_col, sort=True)
        .agg(
            sensor=("sensor", "first"),
            acquisition_date=("acquisition_date", "first"),
            season=("season", "first"),
            lakes=(config.lake_column, lambda values: ";".join(sorted(set(map(str, values))))),
            classes=(
                config.label_column,
                lambda values: ";".join(str(int(value)) for value in sorted(set(values))),
            ),
            total_rows=(scene_col, "size"),
        )
        .reset_index()
    )


def _build_manifest(
    combined: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    protected: set[str],
    config: SplitConfig,
) -> pd.DataFrame:
    scene_col = config.scene_column
    manifest = metadata.copy()
    for name, frame in split_frames.items():
        counts = frame.groupby(scene_col).size()
        manifest[f"{name}_rows"] = manifest[scene_col].map(counts).fillna(0).astype(int)
    manifest["scene_partition"] = np.where(manifest["los_test_rows"] > 0, "los_test", "development")
    manifest["protected_from_los"] = manifest[scene_col].isin(protected)
    manifest["acquisition_date"] = manifest["acquisition_date"].dt.strftime("%Y-%m-%d")
    return manifest[
        [
            scene_col,
            "sensor",
            "acquisition_date",
            "season",
            "lakes",
            "classes",
            "scene_partition",
            "protected_from_los",
            "train_rows",
            "test_rows",
            "los_test_rows",
            "total_rows",
        ]
    ]


def _split_summary(frame: pd.DataFrame, total_rows: int, config: SplitConfig) -> dict[str, Any]:
    dates = _acquisition_date(frame[config.scene_column])
    return {
        "rows": len(frame),
        "row_fraction": len(frame) / total_rows,
        "scenes": int(frame[config.scene_column].nunique()),
        "class_counts": frame[config.label_column].astype(int).value_counts().sort_index().to_dict(),
        "lake_counts": frame[config.lake_column].astype(str).value_counts().sort_index().to_dict(),
        "sensor_counts": _sensor(frame[config.scene_column]).value_counts().sort_index().to_dict(),
        "season_counts": _season(dates).value_counts().reindex(SEASONS, fill_value=0).to_dict(),
    }


def _validate_fractions(config: SplitConfig) -> None:
    fractions = (config.train_fraction, config.test_fraction, config.los_fraction)
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("All split fractions must be strictly between zero and one.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to one, got {sum(fractions):.12g}.")
    if config.los_search_iterations <= 0:
        raise ValueError("los_search_iterations must be positive.")
    unknown_allocations = sorted(
        set(config.protected_scene_familiar_test_rows)
        - set(config.protected_train_scene_ids)
    )
    if unknown_allocations:
        raise ValueError(
            "Familiar-test allocations must refer to protected scenes: "
            f"{unknown_allocations}"
        )
    if any(value < 0 for value in config.protected_scene_familiar_test_rows.values()):
        raise ValueError("Protected-scene familiar-test row counts cannot be negative.")


def _validate_input_columns(frames: Sequence[pd.DataFrame], config: SplitConfig) -> None:
    reference = list(frames[0].columns)
    required = {config.scene_column, config.lake_column, config.label_column}
    for index, frame in enumerate(frames):
        if list(frame.columns) != reference:
            raise ValueError(f"Input CSV {index} does not have the same columns in the same order.")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Input CSV {index} is missing required columns: {missing}")
        if frame[config.scene_column].isna().any():
            raise ValueError(f"Input CSV {index} contains missing scene IDs.")


def _restore_output_order(frame: pd.DataFrame, source_columns: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values("_source_order", kind="stable").loc[:, list(source_columns)].reset_index(drop=True)


def _sensor(scene_ids: pd.Series) -> pd.Series:
    sensor = scene_ids.astype(str).str.split("_").str[0]
    invalid = ~sensor.isin(["LC08", "LC09"])
    if invalid.any():
        raise ValueError(f"Unsupported Landsat scene IDs: {scene_ids.loc[invalid].head(5).tolist()}")
    return sensor


def _acquisition_date(scene_ids: pd.Series) -> pd.Series:
    dates = pd.to_datetime(
        scene_ids.astype(str).str.split("_").str[3],
        format="%Y%m%d",
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError(f"Could not parse acquisition date from scene IDs: {scene_ids.loc[dates.isna()].head(5).tolist()}")
    return dates


def _season(dates: pd.Series) -> pd.Series:
    month = dates.dt.month
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        index=dates.index,
        dtype="string",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_config(config: SplitConfig) -> dict[str, Any]:
    data = asdict(config)
    for field_name in (
        "input_csvs",
        "train_csv",
        "test_csv",
        "los_test_csv",
        "manifest_csv",
        "summary_json",
    ):
        value = data[field_name]
        if isinstance(value, list):
            data[field_name] = [_portable_path(Path(item)) for item in value]
        else:
            data[field_name] = _portable_path(Path(value))
    return data


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return str(resolved)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
