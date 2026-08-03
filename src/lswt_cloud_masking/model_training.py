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
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from .features import CLASS_LABELS, DEFAULT_DROP_COLUMNS, prepare_training_frame


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
    scoring: str = "accuracy"
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

    x_train, y_train, x_test, y_test, feature_columns = prepare_training_frame(
        config.train_csv,
        config.test_csv,
        label_column=config.label_column,
        drop_columns=config.drop_columns,
    )

    cv = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    studies: dict[str, optuna.Study | None] = {}
    models: dict[str, Any] = {}

    studies["decision_tree"], models["decision_tree"] = _fit_decision_tree(
        x_train, y_train, cv, config
    )
    studies["random_forest"], models["random_forest"] = _fit_random_forest(
        x_train, y_train, cv, config
    )
    studies["xgboost"], models["xgboost"] = _fit_xgboost(
        x_train, y_train, cv, config
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
            xgb_label_shift=(name == "xgboost"),
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
        "xgboost_label_shift": "XGBoost is trained with labels 0,1,2 and shifted back to 1,2,3 at prediction time.",
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
    cv: StratifiedKFold,
    *,
    xgb_label_shift: bool = False,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """Evaluate one fitted classifier on train/test and cross-validation."""

    y_train_model = y_train - 1 if xgb_label_shift else y_train
    y_test_model = y_test - 1 if xgb_label_shift else y_test

    pred_train_model = model.predict(x_train)
    pred_test_model = model.predict(x_test)
    pred_train = pred_train_model + 1 if xgb_label_shift else pred_train_model
    pred_test = pred_test_model + 1 if xgb_label_shift else pred_test_model

    cv_scores = cross_val_score(
        model,
        x_train,
        y_train_model,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
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
    cv: StratifiedKFold,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, DecisionTreeClassifier]:
    if config.n_trials_dt <= 0:
        model = DecisionTreeClassifier(random_state=config.random_state)
        model.fit(x_train, y_train)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            random_state=config.random_state,
        )
        return float(cross_val_score(model, x_train, y_train, cv=cv, scoring=config.scoring).mean())

    study = optuna.create_study(direction="maximize", study_name="DecisionTree_Optimization")
    study.optimize(objective, n_trials=config.n_trials_dt, n_jobs=config.optuna_n_jobs)
    model = DecisionTreeClassifier(**study.best_params, random_state=config.random_state)
    model.fit(x_train, y_train)
    return study, model


def _fit_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, RandomForestClassifier]:
    if config.n_trials_rf <= 0:
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 6, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
        return float(cross_val_score(model, x_train, y_train, cv=cv, scoring=config.scoring).mean())

    study = optuna.create_study(direction="maximize", study_name="RandomForest_Optimization")
    study.optimize(objective, n_trials=config.n_trials_rf, n_jobs=config.optuna_n_jobs)
    model = RandomForestClassifier(
        **study.best_params,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return study, model


def _fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: TrainingConfig,
) -> tuple[optuna.Study | None, xgb.XGBClassifier]:
    y_train_zero = y_train - 1
    if config.n_trials_xgb <= 0:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(np.unique(y_train_zero)),
            eval_metric="mlogloss",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train_zero)
        return None, model

    def objective(trial: optuna.Trial) -> float:
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 500),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            objective="multi:softprob",
            num_class=len(np.unique(y_train_zero)),
            eval_metric="mlogloss",
            random_state=config.random_state,
            n_jobs=-1,
        )
        return float(
            cross_val_score(
                model,
                x_train,
                y_train_zero,
                cv=cv,
                scoring=config.scoring,
                n_jobs=-1,
            ).mean()
        )

    study = optuna.create_study(direction="maximize", study_name="XGBoost_Optimization")
    study.optimize(objective, n_trials=config.n_trials_xgb, n_jobs=config.optuna_n_jobs)
    model = xgb.XGBClassifier(
        **study.best_params,
        objective="multi:softprob",
        num_class=len(np.unique(y_train_zero)),
        eval_metric="mlogloss",
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train_zero)
    return study, model


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
