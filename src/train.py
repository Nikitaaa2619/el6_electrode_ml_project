from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "el6-matplotlib")
)

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telemetry.csv"
MODEL = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
THRESHOLD_REPORT = REPORTS / "threshold_analysis.csv"
FORECAST_HORIZON_MINUTES = 10

FEATURES = [
    "electrode_diameter_mm",
    "ambient_temp_c",
    "spindle_rpm",
    "load_pct",
    "motor_current_a",
    "vibration_rms",
    "coolant_temp_c",
    "pressure_bar",
    "operating_hours_since_service",
    "tool_wear_pct",
    "bearing_temp_c",
    "spindle_temp_c",
    "temp_rate_c_per_min",
]


def build_threshold_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    """Calculate validation metrics for a readable grid of decision thresholds."""
    rows = []
    # Two-percentage-point steps avoid selecting a threshold on insignificant
    # floating-point differences between macOS and Linux builds.
    for threshold in np.arange(0.06, 0.951, 0.02):
        pred = (proba >= threshold).astype(int)
        matrix = confusion_matrix(y_true, pred, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
                "true_negative": int(true_negative),
            }
        )
    return pd.DataFrame(rows)


def save_threshold_plot(table: pd.DataFrame, selected_threshold: float) -> None:
    """Plot the validation trade-off used to select the warning threshold."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["recall"], label="Recall")
    ax.plot(table["threshold"], table["f1"], label="F1")
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Выбранный порог: {selected_threshold:.2f}",
    )
    ax.set_title("Выбор порога на валидационной части")
    ax.set_xlabel("Порог предупреждения")
    ax.set_ylabel("Значение метрики")
    ax.set_xlim(0.06, 0.94)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_threshold_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")

    # Chronological split: no random shuffle, because this is a time-dependent task.
    # Labels use the next 10 minutes, so gaps keep each label window inside its split.
    timestamps = df["timestamp"].drop_duplicates().sort_values().to_numpy()
    validation_split = int(len(timestamps) * 0.60)
    test_split = int(len(timestamps) * 0.80)

    train_end = timestamps[validation_split - 1]
    validation_start = timestamps[validation_split + FORECAST_HORIZON_MINUTES]
    validation_end = timestamps[test_split - 1]
    test_start = timestamps[test_split + FORECAST_HORIZON_MINUTES]

    train = df[df["timestamp"] <= train_end]
    validation = df[
        (df["timestamp"] >= validation_start)
        & (df["timestamp"] <= validation_end)
    ]
    test = df[df["timestamp"] >= test_start]

    print(
        f"Time split: {len(train):,} train rows, "
        f"{len(validation):,} validation rows, {len(test):,} test rows"
    )
    print(f"Gap before validation: {FORECAST_HORIZON_MINUTES} minutes")
    print(f"Gap before test      : {FORECAST_HORIZON_MINUTES} minutes")

    X_train, y_train = train[FEATURES], train["overheat_next_10min"]
    X_validation = validation[FEATURES]
    y_validation = validation["overheat_next_10min"]
    X_test, y_test = test[FEATURES], test["overheat_next_10min"]

    print(f"Train risk share     : {y_train.mean():.3%}")
    print(f"Validation risk share: {y_validation.mean():.3%}")
    print(f"Test risk share      : {y_test.mean():.3%}")
    print(f"No-risk baseline accuracy: {(y_test == 0).mean():.3f}")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        # One worker keeps tree ordering reproducible across local and cloud builds.
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    validation_proba = model.predict_proba(X_validation)[:, 1]
    threshold_table = build_threshold_table(y_validation, validation_proba)
    best_row = threshold_table.loc[threshold_table["f1"].idxmax()]
    decision_threshold = float(best_row["threshold"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_REPORT, index=False)
    save_threshold_plot(threshold_table, decision_threshold)

    print("=== Threshold selection on validation ===")
    print(
        f"Selected threshold: {decision_threshold:.2f} | "
        f"precision={best_row['precision']:.3f}, "
        f"recall={best_row['recall']:.3f}, f1={best_row['f1']:.3f}"
    )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= decision_threshold).astype(int)

    print("=== Final test report ===")
    print(classification_report(y_test, pred, digits=3))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC : {average_precision_score(y_test, proba):.3f}")

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    print("=== Feature importance ===")
    print(importance.head(10).to_string())

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": decision_threshold,
        },
        MODEL,
    )
    print(f"Saved model to {MODEL}")
    print(f"Saved threshold analysis to {THRESHOLD_REPORT}")


if __name__ == "__main__":
    main()
