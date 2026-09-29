from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telemetry.csv"
MODEL = ROOT / "models" / "model.joblib"
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


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")

    # Chronological split: no random shuffle, because this is a time-dependent industrial task.
    # Labels use the next 10 minutes, so the gap prevents the training label window
    # from reaching into the test period.
    split = int(len(df) * 0.80)
    test_start = split + FORECAST_HORIZON_MINUTES
    train = df.iloc[:split]
    test = df.iloc[test_start:]

    print(
        f"Time split: {len(train):,} train rows, "
        f"{FORECAST_HORIZON_MINUTES}-minute gap, {len(test):,} test rows"
    )

    X_train, y_train = train[FEATURES], train["overheat_next_10min"]
    X_test, y_test = test[FEATURES], test["overheat_next_10min"]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Classification report ===")
    print(classification_report(y_test, pred, digits=3))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC : {average_precision_score(y_test, proba):.3f}")

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("=== Feature importance ===")
    print(importance.head(10).to_string())

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()
