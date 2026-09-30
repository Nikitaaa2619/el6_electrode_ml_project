from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "el6-matplotlib"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

DATA = ROOT / "data" / "metropt3_minute.csv.gz"
MODEL = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
THRESHOLD_REPORT = REPORTS / "threshold_analysis.csv"
TARGET = "air_leak_condition"
FALSE_POSITIVE_COST = float(os.getenv("FALSE_POSITIVE_COST", "1"))
FALSE_NEGATIVE_COST = float(os.getenv("FALSE_NEGATIVE_COST", "25"))

RAW_FEATURES = [
    "tp2_bar", "tp3_bar", "h1_bar", "dv_pressure_bar", "reservoirs_bar",
    "oil_temperature_c", "motor_current_a", "comp", "dv_electric", "towers",
    "mpg", "lps", "pressure_switch", "oil_level", "caudal_impulses",
]
FEATURES = RAW_FEATURES + [f"{feature}_mean_60m" for feature in RAW_FEATURES] + [
    f"{feature}_std_60m" for feature in RAW_FEATURES
]


def build_threshold_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in np.arange(0.001, 0.201, 0.001):
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(threshold), 3),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "false_positive": int(fp), "false_negative": int(fn),
            "true_positive": int(tp), "true_negative": int(tn),
            "expected_cost": float(fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST),
        })
    return pd.DataFrame(rows)


def save_threshold_plot(table: pd.DataFrame, selected_threshold: float) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["recall"], label="Recall")
    ax.plot(table["threshold"], table["f1"], label="F1")
    ax.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Selected threshold: {selected_threshold:.3f}")
    ax.set(title="Threshold selection on the validation period",
           xlabel="Alert threshold", ylabel="Metric")
    ax.set_xlim(0.0, 0.20)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_threshold_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    pred = (proba >= threshold).astype(int)
    return {
        "test_accuracy": accuracy_score(y_true, pred),
        "test_precision": precision_score(y_true, pred, zero_division=0),
        "test_recall": recall_score(y_true, pred, zero_division=0),
        "test_f1": f1_score(y_true, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, proba),
        "test_pr_auc": average_precision_score(y_true, proba),
    }, pred


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]
    print(f"MetroPT-3 minute rows: {len(train):,} train, {len(validation):,} validation, {len(test):,} test")
    print(f"Positive share: {train[TARGET].mean():.3%} / {validation[TARGET].mean():.3%} / {test[TARGET].mean():.3%}")

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train[TARGET])

    validation_proba = model.predict_proba(validation[FEATURES])[:, 1]
    threshold_table = build_threshold_table(validation[TARGET], validation_proba)
    best = threshold_table.sort_values(["expected_cost", "f1"], ascending=[True, False]).iloc[0]
    threshold = float(best["threshold"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_REPORT, index=False)
    save_threshold_plot(threshold_table, threshold)

    test_proba = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics, test_pred = evaluate(test[TARGET], test_proba, threshold)
    test_metrics["validation_expected_cost"] = float(best["expected_cost"])
    print(f"Threshold {threshold:.3f}; validation recall={best['recall']:.3f}, precision={best['precision']:.3f}, cost={best['expected_cost']:.0f}")
    print(classification_report(test[TARGET], test_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(test[TARGET], test_pred))
    print(f"ROC-AUC: {test_metrics['test_roc_auc']:.3f}")
    print(f"PR-AUC : {test_metrics['test_pr_auc']:.3f}")

    training_profile = {
        feature: {"mean": float(train[feature].mean()), "std": float(train[feature].std())}
        for feature in FEATURES
    }
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "metropt3-air-leak"))
    with mlflow.start_run(run_name="metropt3-random-forest-cost-sensitive") as run:
        mlflow.log_params({
            "dataset": "UCI MetroPT-3", "dataset_doi": "10.24432/C5VW3R",
            "aggregation": "1 minute mean with 60 minute rolling statistics",
            "model": "RandomForestClassifier", "n_estimators": 180,
            "max_depth": 16, "min_samples_leaf": 3,
            "class_weight": "balanced_subsample",
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "decision_threshold": threshold,
        })
        mlflow.log_metrics(test_metrics)
        signature = mlflow.models.infer_signature(
            train[FEATURES].head(5), model.predict_proba(train[FEATURES].head(5))[:, 1]
        )
        mlflow.sklearn.log_model(
            sk_model=model, name="air_leak_condition_model", signature=signature,
            input_example=train[FEATURES].head(3), serialization_format="cloudpickle",
        )
        mlflow.log_artifact(str(THRESHOLD_REPORT), artifact_path="reports")
        mlflow.log_artifact(str(FIGURES / "06_threshold_tradeoff.png"), artifact_path="reports")
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model, "features": FEATURES, "threshold": threshold,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "training_profile": training_profile, "mlflow_run_id": run.info.run_id,
            "dataset": "MetroPT-3", "dataset_doi": "10.24432/C5VW3R",
            "target": TARGET, "raw_features": RAW_FEATURES,
        }, MODEL)
    print(f"Saved model to {MODEL}")
    print(f"MLflow run: {run.info.run_id} ({tracking_uri})")


if __name__ == "__main__":
    main()
