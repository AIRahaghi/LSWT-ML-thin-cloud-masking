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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.base import clone
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
    label_column: str = "lst_class"
    drop_columns: list[str] = field(default_factory=lambda: list(DEFAULT_DROP_COLUMNS))
    random_state: int = 42
    cv_splits: int = 5
    scoring: str = "balanced_accuracy"
    group_column: str | None = "scene_id"
    use_grouped_cv: bool = True
    xgb_sample_weight: str | None = "balanced"
    n_trials_dt: int = 80
    n_trials_rf: int = 120
    n_trials_xgb: int = 120
    optuna_n_jobs: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


def run_training_pipeline(config: TrainingConfig) -> dict[str, Any]:
    """Tune, fit, evaluate, and save DT, RF, and XGBoost classifiers."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test, feature_columns, train_groups, test_groups = (
        _prepare_training_data(config)
    )

    cv, cv_kind = _make_cv(config, train_groups)

    studies: dict[str, optuna.Study | None] = {}
    models: dict[str, Any] = {}

    studies["decision_tree"], models["decision_tree"] = _fit_decision_tree(
        x_train, y_train, cv, train_groups, config
    )
    studies["random_forest"], models["random_forest"] = _fit_random_forest(
        x_train, y_train, cv, train_groups, config
    )
    studies["xgboost"], models["xgboost"] = _fit_xgboost(
        x_train, y_train, cv, train_groups, config
    )

    model_paths = {
        "decision_tree": output_dir / "DT_best_all_general.pkl",
        "random_forest": output_dir / "RF_best_all_general.pkl",
        "xgboost": output_dir / "XGB_best_all_general.pkl",
    }
    for name, model in models.items():
        joblib.dump(model, model_paths[name])
        if studies[name] is not None:
            joblib.dump(studies[name], output_dir / f"{name}_optuna_study.pkl")

    metrics = {
        name: evaluate_classifier(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            cv,
            train_groups=train_groups,
            xgb_label_shift=(name == "xgboost"),
            sample_weight_mode=config.xgb_sample_weight if name == "xgboost" else None,
            scoring=config.scoring,
        )
        for name, model in models.items()
    }

    for name, model_metrics in metrics.items():
        pd.DataFrame(model_metrics["confusion_matrix"]).to_csv(
            output_dir / f"{name}_confusion_matrix.csv",
            index=False,
        )
        pd.DataFrame(model_metrics["classification_report"]).transpose().to_csv(
            output_dir / f"{name}_classification_report.csv",
        )

    metadata = {
        "config": asdict(config),
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,
        "cv_kind": cv_kind,
        "train_groups": _group_summary(train_groups),
        "test_groups": _group_summary(test_groups),
        "train_test_group_overlap": _group_overlap(train_groups, test_groups),
        "xgboost_label_shift": "XGBoost is trained with labels 0,1,2 and shifted back to 1,2,3 at prediction time.",
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

    return {"models": models, "studies": studies, "metrics": metrics, "metadata": metadata}


def evaluate_classifier(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    *,
    train_groups: pd.Series | None = None,
    xgb_label_shift: bool = False,
    sample_weight_mode: str | None = None,
    scoring: str = "balanced_accuracy",
) -> dict[str, Any]:
    """Evaluate one fitted classifier on train/test and cross-validation."""

    y_train_model = y_train - 1 if xgb_label_shift else y_train
    y_test_model = y_test - 1 if xgb_label_shift else y_test

    pred_train_model = model.predict(x_train)
    pred_test_model = model.predict(x_test)
    pred_train = pred_train_model + 1 if xgb_label_shift else pred_train_model
    pred_test = pred_test_model + 1 if xgb_label_shift else pred_test_model

    cv_scores = _cross_val_scores(
        model,
        x_train,
        y_train_model,
        cv=cv,
        scoring=scoring,
        groups=train_groups,
        sample_weight_mode=sample_weight_mode,
    )

    labels = [1, 2, 3]
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    binary = _binary_cloud_metrics(y_test.to_numpy(), np.asarray(pred_test))
    per_class = _producer_user_accuracy(cm, labels)

    return {
        "train_accuracy": accuracy_score(y_train, pred_train),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "train_balanced_accuracy": balanced_accuracy_score(y_train, pred_train),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "classification_report": classification_report(
            y_test,
            pred_test,
            labels=labels,
            target_names=[CLASS_LABELS[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm,
        "producer_user_accuracy": per_class,
        "binary_cloud_accuracy": binary,
    }


def _fit_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, DecisionTreeClassifier]:
    if config.n_trials_dt <= 0:
        model = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=config.random_state,
        )
        model.fit(x_train, y_train)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            random_state=config.random_state,
        )
        return float(_cross_val_scores(model, x_train, y_train, cv, config.scoring, groups).mean())

    study = optuna.create_study(direction="maximize", study_name="DecisionTree_Optimization")
    study.optimize(objective, n_trials=config.n_trials_dt, n_jobs=config.optuna_n_jobs)
    model = DecisionTreeClassifier(**study.best_params, random_state=config.random_state)
    model.fit(x_train, y_train)
    return study, model


def _fit_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, RandomForestClassifier]:
    if config.n_trials_rf <= 0:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features=0.5,
            criterion="entropy",
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1000, step=100),
            max_depth=trial.suggest_categorical("max_depth", [15, 20, 30, 40, None]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            random_state=config.random_state,
            n_jobs=-1,
        )
        return float(_cross_val_scores(model, x_train, y_train, cv, config.scoring, groups).mean())

    study = optuna.create_study(direction="maximize", study_name="RandomForest_Optimization")
    study.optimize(objective, n_trials=config.n_trials_rf, n_jobs=config.optuna_n_jobs)
    model = RandomForestClassifier(
        **study.best_params,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return study, model


def _fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    groups: pd.Series | None,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, xgb.XGBClassifier]:
    y_train_zero = y_train - 1
    final_sample_weight = _sample_weight(y_train_zero, config.xgb_sample_weight)
    if config.n_trials_xgb <= 0:
        model = xgb.XGBClassifier(
            n_estimators=700,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3.0,
            gamma=0.05,
            reg_alpha=1e-6,
            reg_lambda=5.0,
            objective="multi:softprob",
            num_class=len(np.unique(y_train_zero)),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train_zero, sample_weight=final_sample_weight)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 0.5),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            objective="multi:softprob",
            num_class=len(np.unique(y_train_zero)),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=-1,
        )
        return float(
            _cross_val_scores(
                model,
                x_train,
                y_train_zero,
                cv,
                config.scoring,
                groups,
                sample_weight_mode=config.xgb_sample_weight,
            ).mean()
        )

    study = optuna.create_study(direction="maximize", study_name="XGBoost_Optimization")
    study.optimize(objective, n_trials=config.n_trials_xgb, n_jobs=config.optuna_n_jobs)
    model = xgb.XGBClassifier(
        **study.best_params,
        objective="multi:softprob",
        num_class=len(np.unique(y_train_zero)),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train_zero, sample_weight=final_sample_weight)
    return study, model


def _prepare_training_data(
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], pd.Series | None, pd.Series | None]:
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    group_column = config.group_column
    if group_column and (group_column not in train_df.columns or group_column not in test_df.columns):
        group_column = None

    drop_columns = set(config.drop_columns)
    if group_column:
        drop_columns.add(group_column)

    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        y = pd.to_numeric(df[config.label_column], errors="coerce")
        groups = df[group_column].astype(str) if group_column else None
        x = df.drop(columns=[config.label_column]).drop(columns=list(drop_columns), errors="ignore")
        x = x.apply(pd.to_numeric, errors="coerce")
        keep = y.notna() & x.notna().all(axis=1)
        x_out = x.loc[keep].copy()
        y_out = y.loc[keep].astype(int).copy()
        groups_out = groups.loc[keep].copy() if groups is not None else None
        return x_out, y_out, groups_out

    x_train, y_train, train_groups = _split(train_df)
    x_test, y_test, test_groups = _split(test_df)
    feature_columns = list(x_train.columns)

    missing_test = [col for col in feature_columns if col not in x_test.columns]
    if missing_test:
        raise ValueError(f"Test CSV is missing feature columns found in training CSV: {missing_test}")

    return x_train, y_train, x_test[feature_columns], y_test, feature_columns, train_groups, test_groups


def _make_cv(
    config: TrainingConfig,
    groups: pd.Series | None,
) -> tuple[StratifiedKFold | StratifiedGroupKFold, str]:
    if config.use_grouped_cv and groups is not None and groups.nunique() >= config.cv_splits:
        return (
            StratifiedGroupKFold(
                n_splits=config.cv_splits,
                shuffle=True,
                random_state=config.random_state,
            ),
            f"StratifiedGroupKFold grouped by {config.group_column}",
        )
    return (
        StratifiedKFold(
            n_splits=config.cv_splits,
            shuffle=True,
            random_state=config.random_state,
        ),
        "StratifiedKFold",
    )


def _cross_val_scores(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | StratifiedGroupKFold,
    scoring: str,
    groups: pd.Series | None = None,
    sample_weight_mode: str | None = None,
) -> np.ndarray:
    scores: list[float] = []
    split_groups = groups if isinstance(cv, StratifiedGroupKFold) else None

    for train_index, valid_index in cv.split(x, y, split_groups):
        fitted = clone(model)
        x_train_fold = x.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        fit_kwargs = {}
        weights = _sample_weight(y_train_fold, sample_weight_mode)
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        fitted.fit(x_train_fold, y_train_fold, **fit_kwargs)
        pred = fitted.predict(x.iloc[valid_index])
        scores.append(_score_predictions(y.iloc[valid_index], pred, scoring))

    return np.asarray(scores, dtype="float64")


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


def _producer_user_accuracy(cm: np.ndarray, labels: list[int]) -> dict[str, dict[str, float | None]]:
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


def _group_summary(groups: pd.Series | None) -> dict[str, int | None]:
    if groups is None:
        return {"n_groups": None}
    return {
        "n_groups": int(groups.nunique()),
        "n_rows_with_groups": int(groups.notna().sum()),
    }


def _group_overlap(
    train_groups: pd.Series | None,
    test_groups: pd.Series | None,
) -> dict[str, int | list[str] | None]:
    if train_groups is None or test_groups is None:
        return {"n_overlap": None, "examples": None}
    overlap = sorted(set(train_groups.astype(str)).intersection(set(test_groups.astype(str))))
    return {
        "n_overlap": len(overlap),
        "examples": overlap[:10],
    }


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
