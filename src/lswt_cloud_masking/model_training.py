"""Training, tuning, and assessment for the three ML cloud classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from .features import CLASS_LABELS, DEFAULT_DROP_COLUMNS


@dataclass
class TrainingConfig:
    """Configuration for model tuning and assessment."""

    train_csv: str
    test_csv: str
    output_dir: str
    los_test_csv: str | None = None
    label_column: str = "lst_class"
    drop_columns: list[str] = field(default_factory=lambda: list(DEFAULT_DROP_COLUMNS))
    random_state: int = 42
    optuna_seed: int | None = 42
    cv_splits: int = 5
    pixel_cv_splits: int = 5
    scoring: str = "balanced_accuracy"
    selection_scoring: str = "balanced_accuracy"
    group_column: str | None = "scene_id"
    use_grouped_cv: bool = True
    require_group_column: bool = True
    evaluate_pixel_cv: bool = True
    xgb_sample_weight: str | None = "balanced"
    xgb_multi_objective: bool = True
    xgb_selection_group_weight: float = 0.5
    xgb_max_estimators: int = 3000
    xgb_early_stopping_rounds: int = 100
    tuning_cv_seeds: list[int] = field(default_factory=lambda: [42, 142, 242])
    top_candidates_per_run: int = 8
    stability_penalty: float = 0.25
    n_trials_dt: int = 160
    n_trials_rf: int = 150
    n_trials_xgb: int = 150
    optuna_n_jobs: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


@dataclass
class PreparedTrainingData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    x_los_test: pd.DataFrame | None
    y_los_test: pd.Series | None
    feature_columns: list[str]
    train_groups: pd.Series | None
    test_groups: pd.Series | None
    los_test_groups: pd.Series | None


def run_training_pipeline(config: TrainingConfig) -> dict[str, Any]:
    """Tune, fit, evaluate, and save DT, RF, and XGBoost classifiers."""

    _validate_training_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_training_data(config)

    grouped_cvs = [
        _make_grouped_cv(config, data.train_groups, random_state=seed)
        for seed in config.tuning_cv_seeds
    ]
    grouped_cvs = [cv for cv in grouped_cvs if cv is not None]
    pixel_cvs = [
        _make_pixel_cv(config, random_state=seed)
        for seed in config.tuning_cv_seeds
    ]
    grouped_cv = grouped_cvs[0] if grouped_cvs else None
    pixel_cv = pixel_cvs[0]
    _, cv_kind = _make_primary_cv(config, grouped_cv, pixel_cv)

    studies: dict[str, list[optuna.Study]] = {}
    models: dict[str, Any] = {}
    selection: dict[str, Any] = {}

    studies["decision_tree"], models["decision_tree"], selection["decision_tree"] = (
        _fit_decision_tree(
            data.x_train,
            data.y_train,
            grouped_cvs,
            data.train_groups,
            config,
        )
    )
    studies["random_forest"], models["random_forest"], selection["random_forest"] = (
        _fit_random_forest(
            data.x_train,
            data.y_train,
            grouped_cvs,
            data.train_groups,
            config,
        )
    )
    studies["xgboost"], models["xgboost"], selection["xgboost"] = _fit_xgboost(
        data.x_train,
        data.y_train,
        grouped_cvs,
        pixel_cvs,
        data.train_groups,
        config,
    )

    model_paths = {
        "decision_tree": output_dir / "DT_best_all_general.pkl",
        "random_forest": output_dir / "RF_best_all_general.pkl",
        "xgboost": output_dir / "XGB_best_all_general.pkl",
    }
    for name, model in models.items():
        joblib.dump(model, model_paths[name])
        if studies[name]:
            joblib.dump(studies[name], output_dir / f"{name}_optuna_studies.pkl")
            # Preserve the historical single-study filename for existing tooling.
            joblib.dump(studies[name][0], output_dir / f"{name}_optuna_study.pkl")
            for run_index, study in enumerate(studies[name], start=1):
                joblib.dump(
                    study,
                    output_dir / f"{name}_optuna_study_run_{run_index}.pkl",
                )
        _save_candidate_ranking(output_dir, name, selection[name])

    metrics = {
        name: evaluate_classifier(
            model,
            data.x_train,
            data.y_train,
            data.x_test,
            data.y_test,
            grouped_cv=grouped_cv,
            pixel_cv=pixel_cv if config.evaluate_pixel_cv else None,
            train_groups=data.train_groups,
            x_los_test=data.x_los_test,
            y_los_test=data.y_los_test,
            xgb_label_shift=(name == "xgboost"),
            sample_weight_mode=config.xgb_sample_weight if name == "xgboost" else None,
            primary_cv_name="scene_grouped" if config.use_grouped_cv else "pixel_stratified",
        )
        for name, model in models.items()
    }
    for name, model_metrics in metrics.items():
        model_selection = selection[name]
        if model_selection.get("tuned"):
            model_metrics["repeated_grouped_selection"] = {
                "scoring": config.selection_scoring,
                "cv_seeds": config.tuning_cv_seeds,
                "folds_per_seed": config.cv_splits,
                "mean": model_selection["selected_mean_score"],
                "std": model_selection["selected_std_score"],
                "stability_penalty": config.stability_penalty,
                "stability_score": model_selection["selected_stability_score"],
                "fold_scores": model_selection["selected_fold_scores"],
            }

    for name, model_metrics in metrics.items():
        _save_evaluation_tables(output_dir, name, model_metrics)

    metadata = {
        "config": asdict(config),
        "feature_columns": data.feature_columns,
        "class_labels": CLASS_LABELS,
        "primary_cv": cv_kind,
        "robust_selection": {
            "cv_seeds": config.tuning_cv_seeds,
            "folds_per_seed": config.cv_splits,
            "candidate_limit_per_run": config.top_candidates_per_run,
            "criterion": (
                f"mean repeated scene-grouped {config.selection_scoring} - "
                f"{config.stability_penalty} * fold standard deviation"
            ),
            "los_used_for_selection": False,
        },
        "available_cross_validation": [
            value
            for value, enabled in [
                ("scene_grouped", grouped_cv is not None),
                ("pixel_stratified", config.evaluate_pixel_cv),
            ]
            if enabled
        ],
        "datasets": {
            "train": _dataset_summary(data.y_train, data.train_groups),
            "test": _dataset_summary(data.y_test, data.test_groups),
            "los_test": _dataset_summary(data.y_los_test, data.los_test_groups),
        },
        "scene_overlap": {
            "train_test": _group_overlap(data.train_groups, data.test_groups),
            "train_los_test": _group_overlap(data.train_groups, data.los_test_groups),
            "test_los_test": _group_overlap(data.test_groups, data.los_test_groups),
        },
        "model_selection": selection,
        "xgboost_label_shift": (
            "XGBoost is trained with labels 0,1,2 and shifted back to 1,2,3 at prediction time."
        ),
        "xgboost_sample_weight": config.xgb_sample_weight,
        "model_paths": {name: str(path) for name, path in model_paths.items()},
    }

    (output_dir / "model_metadata.json").write_text(
        json.dumps(_to_jsonable(metadata), indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(_to_jsonable(metrics), indent=2),
        encoding="utf-8",
    )

    return {
        "models": models,
        "studies": studies,
        "metrics": metrics,
        "metadata": metadata,
    }


def evaluate_classifier(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    grouped_cv: StratifiedGroupKFold | None,
    pixel_cv: StratifiedKFold | None,
    train_groups: pd.Series | None = None,
    x_los_test: pd.DataFrame | None = None,
    y_los_test: pd.Series | None = None,
    xgb_label_shift: bool = False,
    sample_weight_mode: str | None = None,
    primary_cv_name: str = "scene_grouped",
) -> dict[str, Any]:
    """Evaluate a fitted classifier on train, familiar test, LOS test, and CV."""

    datasets = {
        "train": _evaluate_dataset(model, x_train, y_train, xgb_label_shift),
        "test": _evaluate_dataset(model, x_test, y_test, xgb_label_shift),
    }
    if x_los_test is not None and y_los_test is not None:
        datasets["los_test"] = _evaluate_dataset(
            model,
            x_los_test,
            y_los_test,
            xgb_label_shift,
        )

    y_model = y_train - 1 if xgb_label_shift else y_train
    cross_validation: dict[str, Any] = {}
    if grouped_cv is not None:
        cross_validation["scene_grouped"] = _cross_val_metrics(
            model,
            x_train,
            y_model,
            grouped_cv,
            groups=train_groups,
            sample_weight_mode=sample_weight_mode,
        )
    if pixel_cv is not None:
        cross_validation["pixel_stratified"] = _cross_val_metrics(
            model,
            x_train,
            y_model,
            pixel_cv,
            groups=None,
            sample_weight_mode=sample_weight_mode,
        )

    primary = cross_validation[primary_cv_name]
    result = {
        "datasets": datasets,
        "cross_validation": cross_validation,
        # Backward-compatible summary fields used by older notebooks/scripts.
        "train_accuracy": datasets["train"]["accuracy"],
        "test_accuracy": datasets["test"]["accuracy"],
        "train_balanced_accuracy": datasets["train"]["balanced_accuracy"],
        "test_balanced_accuracy": datasets["test"]["balanced_accuracy"],
        "cv_mean": primary["balanced_accuracy"]["mean"],
        "cv_std": primary["balanced_accuracy"]["std"],
        "classification_report": datasets["test"]["classification_report"],
        "confusion_matrix": datasets["test"]["confusion_matrix"],
        "producer_user_accuracy": datasets["test"]["producer_user_accuracy"],
        "binary_cloud_accuracy": datasets["test"]["binary_cloud_accuracy"],
    }
    if "los_test" in datasets:
        result["los_test_accuracy"] = datasets["los_test"]["accuracy"]
        result["los_test_balanced_accuracy"] = datasets["los_test"]["balanced_accuracy"]
    return result


def _evaluate_dataset(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    xgb_label_shift: bool,
) -> dict[str, Any]:
    pred_model = model.predict(x)
    pred = np.asarray(pred_model) + 1 if xgb_label_shift else np.asarray(pred_model)
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


def _fit_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    grouped_cvs: list[StratifiedGroupKFold],
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[list[optuna.Study], DecisionTreeClassifier, dict[str, Any]]:
    if config.n_trials_dt <= 0:
        model = DecisionTreeClassifier(
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.random_state,
        )
        model.fit(x_train, y_train)
        return [], model, {"tuned": False, "parameters": model.get_params(deep=False)}

    studies: list[optuna.Study] = []
    candidate_records: list[dict[str, Any]] = []
    for run_index, (cv_seed, cv) in enumerate(
        zip(config.tuning_cv_seeds, grouped_cvs, strict=True),
        start=1,
    ):
        def objective(trial: optuna.Trial) -> float:
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 2, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 100),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 100),
                max_features=trial.suggest_categorical(
                    "max_features", [None, "sqrt", "log2", 0.5, 0.75]
                ),
                criterion=trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),
                splitter=trial.suggest_categorical("splitter", ["best", "random"]),
                class_weight=trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                ccp_alpha=trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True),
                random_state=config.random_state,
            )
            return _cross_val_score_mean(
                model,
                x_train,
                y_train,
                cv,
                config.scoring,
                groups,
            )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"DecisionTree_Optimization_Run_{run_index}_CVSeed_{cv_seed}",
            sampler=optuna.samplers.TPESampler(
                seed=_optuna_seed(config, run_index - 1)
            ),
        )
        study.optimize(
            objective,
            n_trials=config.n_trials_dt,
            n_jobs=config.optuna_n_jobs,
        )
        studies.append(study)
        candidate_records.extend(
            _top_study_candidates(study, run_index, cv_seed, config)
        )

    candidates = _merge_candidate_records(candidate_records)
    ranking = _rank_sklearn_candidates(
        candidates,
        "Decision Tree",
        lambda params: DecisionTreeClassifier(
            **params,
            random_state=config.random_state,
        ),
        x_train,
        y_train,
        grouped_cvs,
        groups,
        config,
    )
    selected = ranking[0]
    model = DecisionTreeClassifier(
        **selected["parameters"],
        random_state=config.random_state,
    )
    model.fit(x_train, y_train)
    return studies, model, _robust_selection_metadata(selected, ranking, config)


def _fit_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    grouped_cvs: list[StratifiedGroupKFold],
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[list[optuna.Study], RandomForestClassifier, dict[str, Any]]:
    if config.n_trials_rf <= 0:
        model = RandomForestClassifier(
            n_estimators=700,
            max_depth=30,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features=0.5,
            criterion="entropy",
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return [], model, {"tuned": False, "parameters": model.get_params(deep=False)}

    studies: list[optuna.Study] = []
    candidate_records: list[dict[str, Any]] = []
    for run_index, (cv_seed, cv) in enumerate(
        zip(config.tuning_cv_seeds, grouped_cvs, strict=True),
        start=1,
    ):
        def objective(trial: optuna.Trial) -> float:
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 500, 1500, step=100),
                max_depth=trial.suggest_categorical(
                    "max_depth", [10, 15, 20, 30, 40, None]
                ),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                max_features=trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
                ),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                criterion=trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),
                class_weight=trial.suggest_categorical(
                    "class_weight", [None, "balanced", "balanced_subsample"]
                ),
                random_state=config.random_state,
                n_jobs=-1,
            )
            return _cross_val_score_mean(
                model,
                x_train,
                y_train,
                cv,
                config.scoring,
                groups,
            )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"RandomForest_Optimization_Run_{run_index}_CVSeed_{cv_seed}",
            sampler=optuna.samplers.TPESampler(
                seed=_optuna_seed(config, run_index - 1)
            ),
        )
        study.optimize(
            objective,
            n_trials=config.n_trials_rf,
            n_jobs=config.optuna_n_jobs,
        )
        studies.append(study)
        candidate_records.extend(
            _top_study_candidates(study, run_index, cv_seed, config)
        )

    candidates = _merge_candidate_records(candidate_records)
    ranking = _rank_sklearn_candidates(
        candidates,
        "Random Forest",
        lambda params: RandomForestClassifier(
            **params,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        x_train,
        y_train,
        grouped_cvs,
        groups,
        config,
    )
    selected = ranking[0]
    model = RandomForestClassifier(
        **selected["parameters"],
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return studies, model, _robust_selection_metadata(selected, ranking, config)


def _fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    grouped_cvs: list[StratifiedGroupKFold],
    pixel_cvs: list[StratifiedKFold],
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[list[optuna.Study], xgb.XGBClassifier, dict[str, Any]]:
    y_zero = y_train - 1
    final_sample_weight = _sample_weight(y_zero, config.xgb_sample_weight)
    if config.n_trials_xgb <= 0:
        params = {
            "n_estimators": 1100,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "min_child_weight": 2.0,
            "gamma": 0.02,
            "reg_alpha": 1e-6,
            "reg_lambda": 1.0,
        }
        model = _new_xgb_model(params, config)
        model.fit(x_train, y_zero, sample_weight=final_sample_weight)
        return [], model, {"tuned": False, "parameters": params}

    if config.xgb_multi_objective and not grouped_cvs:
        raise ValueError("XGBoost multi-objective tuning requires scene-grouped CV.")

    studies: list[optuna.Study] = []
    candidate_records: list[dict[str, Any]] = []
    for run_index, (cv_seed, grouped_cv, pixel_cv) in enumerate(
        zip(config.tuning_cv_seeds, grouped_cvs, pixel_cvs, strict=True),
        start=1,
    ):
        def objective(trial: optuna.Trial) -> float | tuple[float, float]:
            params = _suggest_xgb_params(trial)
            grouped_scores, grouped_iterations = _xgb_early_stopped_cv(
                params,
                x_train,
                y_zero,
                grouped_cv,
                groups,
                config,
            )
            grouped_score = float(np.mean(grouped_scores))
            trial.set_user_attr("scene_grouped_score", grouped_score)
            trial.set_user_attr(
                "scene_grouped_best_iterations",
                grouped_iterations,
            )

            if config.xgb_multi_objective:
                pixel_scores, pixel_iterations = _xgb_early_stopped_cv(
                    params,
                    x_train,
                    y_zero,
                    pixel_cv,
                    None,
                    config,
                )
                pixel_score = float(np.mean(pixel_scores))
                trial.set_user_attr("pixel_stratified_score", pixel_score)
                trial.set_user_attr(
                    "pixel_stratified_best_iterations",
                    pixel_iterations,
                )
                return grouped_score, pixel_score
            return grouped_score

        sampler = optuna.samplers.TPESampler(
            seed=_optuna_seed(config, run_index - 1)
        )
        if config.xgb_multi_objective:
            study = optuna.create_study(
                directions=["maximize", "maximize"],
                study_name=f"XGBoost_MultiObjective_Run_{run_index}_CVSeed_{cv_seed}",
                sampler=sampler,
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                study_name=f"XGBoost_Optimization_Run_{run_index}_CVSeed_{cv_seed}",
                sampler=sampler,
            )
        study.optimize(
            objective,
            n_trials=config.n_trials_xgb,
            n_jobs=config.optuna_n_jobs,
        )
        studies.append(study)
        candidate_records.extend(
            _top_study_candidates(
                study,
                run_index,
                cv_seed,
                config,
                xgboost=True,
            )
        )

    candidates = _merge_candidate_records(candidate_records)
    ranking = _rank_xgb_candidates(
        candidates,
        "XGBoost",
        x_train,
        y_zero,
        grouped_cvs,
        groups,
        config,
    )
    selected = ranking[0]
    final_n_estimators = selected["final_n_estimators"]
    final_params = dict(selected["parameters"])
    final_params["n_estimators"] = final_n_estimators
    model = _new_xgb_model(final_params, config)
    model.fit(x_train, y_zero, sample_weight=final_sample_weight)

    metadata = _robust_selection_metadata(selected, ranking, config)
    metadata.update({
        "multi_objective": config.xgb_multi_objective,
        "tuning_objectives": [
            f"scene_grouped_{config.scoring}",
            f"pixel_stratified_{config.scoring}",
        ]
        if config.xgb_multi_objective
        else [config.scoring],
        "candidate_prescreen_rule": (
            "weighted harmonic mean of grouped and pixel-stratified tuning objectives"
            if config.xgb_multi_objective
            else "grouped-CV tuning objective"
        ),
        "candidate_prescreen_group_weight": config.xgb_selection_group_weight,
        "early_stopping_rounds": config.xgb_early_stopping_rounds,
        "maximum_estimators_during_tuning": config.xgb_max_estimators,
        "fold_best_iterations": selected["fold_best_iterations"],
        "final_n_estimators": final_n_estimators,
        "parameters": final_params,
    })
    return studies, model, metadata


def _suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
    }


def _new_xgb_model(
    params: dict[str, Any],
    config: TrainingConfig,
    *,
    early_stopping: bool = False,
) -> xgb.XGBClassifier:
    model_params = {
        **params,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": config.random_state,
        "n_jobs": -1,
    }
    if early_stopping:
        model_params["n_estimators"] = config.xgb_max_estimators
        model_params["early_stopping_rounds"] = config.xgb_early_stopping_rounds
    return xgb.XGBClassifier(**model_params)


def _xgb_early_stopped_cv(
    params: dict[str, Any],
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    groups: pd.Series | None,
    config: TrainingConfig,
    *,
    scoring: str | None = None,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    best_iterations: list[int] = []
    split_groups = groups if isinstance(cv, StratifiedGroupKFold) else None
    for train_index, valid_index in cv.split(x, y, split_groups):
        model = _new_xgb_model(params, config, early_stopping=True)
        y_fold = y.iloc[train_index]
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(x.iloc[valid_index], y.iloc[valid_index])],
            "verbose": False,
        }
        weights = _sample_weight(y_fold, config.xgb_sample_weight)
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
            fit_kwargs["sample_weight_eval_set"] = [
                _sample_weight(y.iloc[valid_index], config.xgb_sample_weight)
            ]
        model.fit(x.iloc[train_index], y_fold, **fit_kwargs)
        pred = model.predict(x.iloc[valid_index])
        scores.append(
            _score_predictions(
                y.iloc[valid_index],
                pred,
                config.scoring if scoring is None else scoring,
            )
        )
        best_iteration = getattr(model, "best_iteration", None)
        best_iterations.append(
            config.xgb_max_estimators if best_iteration is None else int(best_iteration) + 1
        )
    return scores, best_iterations


def _top_study_candidates(
    study: optuna.Study,
    run_index: int,
    cv_seed: int,
    config: TrainingConfig,
    *,
    xgboost: bool = False,
) -> list[dict[str, Any]]:
    ranked_trials: list[tuple[float, optuna.trial.FrozenTrial]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.values is None:
            continue
        values = [float(value) for value in trial.values]
        if xgboost and config.xgb_multi_objective:
            prescreen_score = _weighted_harmonic_mean(
                values[0],
                values[1],
                config.xgb_selection_group_weight,
            )
        else:
            prescreen_score = values[0]
        ranked_trials.append((prescreen_score, trial))

    ranked_trials.sort(key=lambda item: (-item[0], item[1].number))
    selected = ranked_trials[: min(config.top_candidates_per_run, len(ranked_trials))]
    if not selected:
        raise RuntimeError(f"Optuna study '{study.study_name}' produced no completed trials.")
    return [
        {
            "parameters": dict(trial.params),
            "sources": [
                {
                    "run": run_index,
                    "cv_seed": cv_seed,
                    "trial": trial.number,
                    "tuning_values": [float(value) for value in trial.values or ()],
                    "prescreen_score": float(score),
                }
            ],
        }
        for score, trial in selected
    ]


def _merge_candidate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        key = json.dumps(record["parameters"], sort_keys=True, separators=(",", ":"))
        if key not in merged:
            merged[key] = {
                "candidate_id": len(merged) + 1,
                "parameters": record["parameters"],
                "sources": [],
            }
        merged[key]["sources"].extend(record["sources"])
    return list(merged.values())


def _rank_sklearn_candidates(
    candidates: list[dict[str, Any]],
    model_name: str,
    model_factory: Any,
    x: pd.DataFrame,
    y: pd.Series,
    grouped_cvs: list[StratifiedGroupKFold],
    groups: pd.Series | None,
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            f"{model_name}: repeated grouped-CV candidate "
            f"{candidate_index}/{len(candidates)}",
            flush=True,
        )
        fold_scores: list[float] = []
        scores_by_seed: dict[str, list[float]] = {}
        for cv_seed, cv in zip(config.tuning_cv_seeds, grouped_cvs, strict=True):
            seed_scores = _cross_val_scores(
                model_factory(candidate["parameters"]),
                x,
                y,
                cv,
                config.selection_scoring,
                groups,
            )
            scores_by_seed[str(cv_seed)] = seed_scores
            fold_scores.extend(seed_scores)
        evaluated.append(
            _candidate_evaluation(candidate, fold_scores, scores_by_seed, config)
        )
    return _sort_candidate_ranking(evaluated)


def _rank_xgb_candidates(
    candidates: list[dict[str, Any]],
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    grouped_cvs: list[StratifiedGroupKFold],
    groups: pd.Series | None,
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            f"{model_name}: repeated grouped-CV candidate "
            f"{candidate_index}/{len(candidates)}",
            flush=True,
        )
        fold_scores: list[float] = []
        best_iterations: list[int] = []
        scores_by_seed: dict[str, list[float]] = {}
        iterations_by_seed: dict[str, list[int]] = {}
        for cv_seed, cv in zip(config.tuning_cv_seeds, grouped_cvs, strict=True):
            seed_scores, seed_iterations = _xgb_early_stopped_cv(
                candidate["parameters"],
                x,
                y,
                cv,
                groups,
                config,
                scoring=config.selection_scoring,
            )
            scores_by_seed[str(cv_seed)] = seed_scores
            iterations_by_seed[str(cv_seed)] = seed_iterations
            fold_scores.extend(seed_scores)
            best_iterations.extend(seed_iterations)
        evaluation = _candidate_evaluation(
            candidate,
            fold_scores,
            scores_by_seed,
            config,
        )
        evaluation["fold_best_iterations"] = best_iterations
        evaluation["best_iterations_by_seed"] = iterations_by_seed
        evaluation["final_n_estimators"] = int(
            np.clip(
                round(float(np.median(best_iterations))),
                1,
                config.xgb_max_estimators,
            )
        )
        evaluated.append(evaluation)
    return _sort_candidate_ranking(evaluated)


def _candidate_evaluation(
    candidate: dict[str, Any],
    fold_scores: list[float],
    scores_by_seed: dict[str, list[float]],
    config: TrainingConfig,
) -> dict[str, Any]:
    values = np.asarray(fold_scores, dtype="float64")
    mean_score = float(values.mean())
    std_score = float(values.std())
    return {
        **candidate,
        "fold_scores": [float(value) for value in fold_scores],
        "fold_scores_by_seed": scores_by_seed,
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": float(values.min()),
        "max_score": float(values.max()),
        "stability_score": mean_score - config.stability_penalty * std_score,
    }


def _sort_candidate_ranking(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        raise RuntimeError("No candidate parameter sets were available for robust selection.")
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["stability_score"],
            -candidate["mean_score"],
            candidate["std_score"],
            candidate["candidate_id"],
        ),
    )
    for rank, candidate in enumerate(ranking, start=1):
        candidate["rank"] = rank
    return ranking


def _robust_selection_metadata(
    selected: dict[str, Any],
    ranking: list[dict[str, Any]],
    config: TrainingConfig,
) -> dict[str, Any]:
    return {
        "tuned": True,
        "selection_rule": (
            f"mean repeated scene-grouped {config.selection_scoring} - "
            f"{config.stability_penalty} * fold standard deviation"
        ),
        "tuning_runs": len(config.tuning_cv_seeds),
        "tuning_cv_seeds": config.tuning_cv_seeds,
        "folds_per_seed": config.cv_splits,
        "top_candidates_per_run": config.top_candidates_per_run,
        "candidate_entries_before_deduplication": sum(
            len(candidate["sources"]) for candidate in ranking
        ),
        "unique_candidates_evaluated": len(ranking),
        "selected_candidate_id": selected["candidate_id"],
        "selected_mean_score": selected["mean_score"],
        "selected_std_score": selected["std_score"],
        "selected_stability_score": selected["stability_score"],
        "selected_fold_scores": selected["fold_scores"],
        "selected_sources": selected["sources"],
        "parameters": selected["parameters"],
        "candidate_ranking": ranking,
    }


def _weighted_harmonic_mean(grouped: float, pixel: float, group_weight: float) -> float:
    if not 0 <= group_weight <= 1:
        raise ValueError("XGBoost group weight must be between zero and one.")
    if grouped <= 0 or pixel <= 0:
        return 0.0
    return float(1.0 / (group_weight / grouped + (1.0 - group_weight) / pixel))


def _prepare_training_data(config: TrainingConfig) -> PreparedTrainingData:
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    los_df = pd.read_csv(config.los_test_csv) if config.los_test_csv else None

    frames = {"train": train_df, "test": test_df}
    if los_df is not None:
        frames["los_test"] = los_df
    for name, frame in frames.items():
        if config.label_column not in frame.columns:
            raise KeyError(f"{name} CSV is missing label column '{config.label_column}'.")

    group_column = config.group_column
    missing_group = group_column is None or any(group_column not in frame.columns for frame in frames.values())
    if missing_group:
        if config.require_group_column or config.use_grouped_cv or config.xgb_multi_objective:
            raise ValueError(
                f"All datasets must contain group column '{config.group_column}' for this configuration."
            )
        group_column = None

    drop_columns = set(config.drop_columns)
    if group_column:
        drop_columns.add(group_column)

    def split(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        y = pd.to_numeric(frame[config.label_column], errors="coerce")
        groups: pd.Series | None = None
        if group_column:
            if frame[group_column].isna().any() or frame[group_column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{name} CSV contains missing or blank group values.")
            groups = frame[group_column].astype(str)
        x = frame.drop(columns=[config.label_column]).drop(columns=list(drop_columns), errors="ignore")
        x = x.apply(pd.to_numeric, errors="coerce")
        keep = y.notna() & x.notna().all(axis=1)
        x_out = x.loc[keep].copy()
        y_out = y.loc[keep].astype(int).copy()
        groups_out = groups.loc[keep].copy() if groups is not None else None
        if x_out.empty:
            raise ValueError(f"{name} CSV contains no complete model rows.")
        unexpected_labels = sorted(set(y_out.unique()) - set(CLASS_LABELS))
        if unexpected_labels:
            raise ValueError(f"{name} CSV contains unexpected labels: {unexpected_labels}")
        return x_out, y_out, groups_out

    x_train, y_train, train_groups = split(train_df, "train")
    x_test, y_test, test_groups = split(test_df, "test")
    if los_df is not None:
        x_los, y_los, los_groups = split(los_df, "los_test")
    else:
        x_los, y_los, los_groups = None, None, None

    feature_columns = list(x_train.columns)
    x_test = _align_features(x_test, feature_columns, "test")
    if x_los is not None:
        x_los = _align_features(x_los, feature_columns, "los_test")

    if los_groups is not None:
        if _group_overlap(train_groups, los_groups)["n_overlap"]:
            raise ValueError("LOS test contains scene IDs found in training.")
        if _group_overlap(test_groups, los_groups)["n_overlap"]:
            raise ValueError("LOS test contains scene IDs found in the familiar test.")

    return PreparedTrainingData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_los_test=x_los,
        y_los_test=y_los,
        feature_columns=feature_columns,
        train_groups=train_groups,
        test_groups=test_groups,
        los_test_groups=los_groups,
    )


def _align_features(x: pd.DataFrame, feature_columns: list[str], name: str) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in x.columns]
    if missing:
        raise ValueError(f"{name} CSV is missing training feature columns: {missing}")
    return x[feature_columns].copy()


def _make_grouped_cv(
    config: TrainingConfig,
    groups: pd.Series | None,
    *,
    random_state: int | None = None,
) -> StratifiedGroupKFold | None:
    if groups is None:
        return None
    if groups.nunique() < config.cv_splits:
        raise ValueError(
            f"Grouped CV needs at least {config.cv_splits} groups; found {groups.nunique()}."
        )
    return StratifiedGroupKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state if random_state is None else random_state,
    )


def _make_pixel_cv(
    config: TrainingConfig,
    *,
    random_state: int | None = None,
) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.pixel_cv_splits,
        shuffle=True,
        random_state=config.random_state if random_state is None else random_state,
    )


def _make_primary_cv(
    config: TrainingConfig,
    grouped_cv: StratifiedGroupKFold | None,
    pixel_cv: StratifiedKFold,
) -> tuple[StratifiedKFold | StratifiedGroupKFold, str]:
    if config.use_grouped_cv:
        if grouped_cv is None:
            raise ValueError("Grouped CV was requested but no valid group column is available.")
        return grouped_cv, f"StratifiedGroupKFold grouped by {config.group_column}"
    return pixel_cv, "StratifiedKFold over pixels"


def _cross_val_score_mean(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    scoring: str,
    groups: pd.Series | None,
    sample_weight_mode: str | None = None,
) -> float:
    return float(
        np.mean(
            _cross_val_scores(
                model,
                x,
                y,
                cv,
                scoring,
                groups,
                sample_weight_mode,
            )
        )
    )


def _cross_val_scores(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    scoring: str,
    groups: pd.Series | None,
    sample_weight_mode: str | None = None,
) -> list[float]:
    split_groups = groups if isinstance(cv, StratifiedGroupKFold) else None
    scores: list[float] = []
    for train_index, valid_index in cv.split(x, y, split_groups):
        fitted = clone(model)
        y_fold = y.iloc[train_index]
        fit_kwargs: dict[str, Any] = {}
        weights = _sample_weight(y_fold, sample_weight_mode)
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        fitted.fit(x.iloc[train_index], y_fold, **fit_kwargs)
        pred = fitted.predict(x.iloc[valid_index])
        scores.append(_score_predictions(y.iloc[valid_index], pred, scoring))
    return scores


def _cross_val_metrics(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    *,
    groups: pd.Series | None,
    sample_weight_mode: str | None,
) -> dict[str, Any]:
    split_groups = groups if isinstance(cv, StratifiedGroupKFold) else None
    pooled = np.empty(len(y), dtype="int64")
    folds: list[dict[str, Any]] = []
    for fold, (train_index, valid_index) in enumerate(cv.split(x, y, split_groups), start=1):
        fitted = clone(model)
        y_fold = y.iloc[train_index]
        fit_kwargs: dict[str, Any] = {}
        weights = _sample_weight(y_fold, sample_weight_mode)
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        fitted.fit(x.iloc[train_index], y_fold, **fit_kwargs)
        pred = np.asarray(fitted.predict(x.iloc[valid_index]), dtype="int64")
        pooled[valid_index] = pred
        folds.append(
            {
                "fold": fold,
                "n_rows": len(valid_index),
                "n_groups": int(groups.iloc[valid_index].nunique())
                if isinstance(cv, StratifiedGroupKFold) and groups is not None
                else None,
                "accuracy": float(accuracy_score(y.iloc[valid_index], pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y.iloc[valid_index], pred)
                ),
                "f1_macro": float(
                    f1_score(y.iloc[valid_index], pred, average="macro", zero_division=0)
                ),
            }
        )

    result: dict[str, Any] = {"folds": folds}
    for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
        values = np.asarray([fold[metric] for fold in folds], dtype="float64")
        if metric == "accuracy":
            pooled_value = accuracy_score(y, pooled)
        elif metric == "balanced_accuracy":
            pooled_value = balanced_accuracy_score(y, pooled)
        else:
            pooled_value = f1_score(y, pooled, average="macro", zero_division=0)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "pooled": float(pooled_value),
        }
    return result


def _sample_weight(y: pd.Series, mode: str | None) -> np.ndarray | None:
    if mode is None:
        return None
    if mode != "balanced":
        raise ValueError(f"Unsupported sample-weight mode: {mode}")
    return compute_sample_weight(class_weight="balanced", y=y)


def _score_predictions(y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if scoring == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(
        f"Unsupported scoring '{scoring}'. Use 'balanced_accuracy', 'accuracy', or 'f1_macro'."
    )


def _save_evaluation_tables(output_dir: Path, name: str, metrics: dict[str, Any]) -> None:
    datasets = metrics["datasets"]
    for dataset_name in ("test", "los_test"):
        if dataset_name not in datasets:
            continue
        dataset = datasets[dataset_name]
        pd.DataFrame(dataset["confusion_matrix"]).to_csv(
            output_dir / f"{name}_{dataset_name}_confusion_matrix.csv",
            index=False,
        )
        pd.DataFrame(dataset["classification_report"]).transpose().to_csv(
            output_dir / f"{name}_{dataset_name}_classification_report.csv"
        )
    # Keep the historical familiar-test filenames for downstream compatibility.
    pd.DataFrame(datasets["test"]["confusion_matrix"]).to_csv(
        output_dir / f"{name}_confusion_matrix.csv",
        index=False,
    )
    pd.DataFrame(datasets["test"]["classification_report"]).transpose().to_csv(
        output_dir / f"{name}_classification_report.csv"
    )


def _save_candidate_ranking(
    output_dir: Path,
    name: str,
    selection: dict[str, Any],
) -> None:
    ranking = selection.get("candidate_ranking")
    if not ranking:
        return
    rows: list[dict[str, Any]] = []
    for candidate in ranking:
        rows.append(
            {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "stability_score": candidate["stability_score"],
                "mean_score": candidate["mean_score"],
                "std_score": candidate["std_score"],
                "min_score": candidate["min_score"],
                "max_score": candidate["max_score"],
                "n_repeated_folds": len(candidate["fold_scores"]),
                "final_n_estimators": candidate.get("final_n_estimators"),
                "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                "source_trials": json.dumps(candidate["sources"], sort_keys=True),
                "fold_scores": json.dumps(candidate["fold_scores"]),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / f"{name}_candidate_ranking.csv",
        index=False,
    )


def _producer_user_accuracy(
    cm: np.ndarray,
    labels: list[int],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        producer = tp / (tp + fn) if (tp + fn) else None
        user = tp / (tp + fp) if (tp + fp) else None
        out[str(label)] = {
            "label": CLASS_LABELS[label],
            "producer_accuracy": producer,
            "user_accuracy": user,
        }
    return out


def _binary_cloud_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    true_cloud = np.isin(y_true, [1, 2]).astype("uint8")
    pred_cloud = np.isin(y_pred, [1, 2]).astype("uint8")
    cm = confusion_matrix(true_cloud, pred_cloud, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": (tp + tn) / cm.sum() if cm.sum() else None,
        "producer_accuracy_cloud": tp / (tp + fn) if (tp + fn) else None,
        "user_accuracy_cloud": tp / (tp + fp) if (tp + fp) else None,
    }


def _dataset_summary(y: pd.Series | None, groups: pd.Series | None) -> dict[str, Any] | None:
    if y is None:
        return None
    return {
        "n_rows": len(y),
        "n_groups": int(groups.nunique()) if groups is not None else None,
        "class_counts": y.value_counts().sort_index().to_dict(),
    }


def _group_overlap(
    first: pd.Series | None,
    second: pd.Series | None,
) -> dict[str, int | list[str] | None]:
    if first is None or second is None:
        return {"n_overlap": None, "examples": None}
    overlap = sorted(set(first.astype(str)).intersection(set(second.astype(str))))
    return {"n_overlap": len(overlap), "examples": overlap[:10]}


def _validate_training_config(config: TrainingConfig) -> None:
    supported_scores = {"accuracy", "balanced_accuracy", "f1_macro"}
    if config.scoring not in supported_scores:
        raise ValueError(f"Unsupported tuning scoring metric: {config.scoring}")
    if config.selection_scoring not in supported_scores:
        raise ValueError(
            f"Unsupported robust-selection scoring metric: {config.selection_scoring}"
        )
    if config.cv_splits < 2 or config.pixel_cv_splits < 2:
        raise ValueError("Cross-validation split counts must be at least two.")
    if not 0 <= config.xgb_selection_group_weight <= 1:
        raise ValueError("xgb_selection_group_weight must be between zero and one.")
    if config.xgb_max_estimators <= 0 or config.xgb_early_stopping_rounds <= 0:
        raise ValueError("XGBoost estimator and early-stopping limits must be positive.")
    if not config.use_grouped_cv and not config.evaluate_pixel_cv:
        raise ValueError("Pixel CV must be evaluated when it is the primary CV strategy.")
    if config.xgb_multi_objective and not config.evaluate_pixel_cv:
        raise ValueError("XGBoost multi-objective tuning requires pixel CV evaluation.")
    if not config.tuning_cv_seeds:
        raise ValueError("At least one tuning CV seed is required.")
    if len(set(config.tuning_cv_seeds)) != len(config.tuning_cv_seeds):
        raise ValueError("tuning_cv_seeds must contain distinct values.")
    if config.top_candidates_per_run <= 0:
        raise ValueError("top_candidates_per_run must be positive.")
    if config.stability_penalty < 0:
        raise ValueError("stability_penalty cannot be negative.")
    if min(config.n_trials_dt, config.n_trials_rf, config.n_trials_xgb) < 0:
        raise ValueError("Optuna trial counts cannot be negative.")
    if any(
        trials > 0
        for trials in (config.n_trials_dt, config.n_trials_rf, config.n_trials_xgb)
    ) and not config.use_grouped_cv:
        raise ValueError("Repeated robust tuning requires use_grouped_cv=true.")


def _optuna_seed(config: TrainingConfig, run_offset: int = 0) -> int:
    base_seed = config.random_state if config.optuna_seed is None else config.optuna_seed
    return base_seed + run_offset


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_to_jsonable(child) for child in value]
    return value
