from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.notifications import AlertNotifier
from app.storage import MonitoringStore

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = joblib.load(ROOT / "models" / "model.joblib")
MODEL = ARTIFACT["model"]
FEATURES = ARTIFACT["features"]
RAW_FEATURES = ARTIFACT.get("raw_features", FEATURES)
DECISION_THRESHOLD = ARTIFACT.get("threshold", 0.5)
TRAINING_PROFILE = ARTIFACT.get("training_profile", {})
DEMO = pd.read_csv(ROOT / "data" / "metropt3_demo.csv")
STORE = MonitoringStore(ROOT)
NOTIFIER = AlertNotifier()

app = FastAPI(title="MetroPT-3 Compressor Condition Monitoring API")


class Telemetry(BaseModel):
    machine_id: str | None = Field(default=None, min_length=1, max_length=50, pattern=r"^[A-Za-z0-9._-]+$")
    tp2_bar: float
    tp3_bar: float
    h1_bar: float
    dv_pressure_bar: float
    reservoirs_bar: float
    oil_temperature_c: float
    motor_current_a: float = Field(..., ge=0)
    comp: float = Field(..., ge=0, le=1)
    dv_electric: float = Field(..., ge=0, le=1)
    towers: float = Field(..., ge=0, le=1)
    mpg: float = Field(..., ge=0, le=1)
    lps: float = Field(..., ge=0, le=1)
    pressure_switch: float = Field(..., ge=0, le=1)
    oil_level: float = Field(..., ge=0, le=1)
    caudal_impulses: float = Field(..., ge=0, le=1)
    rolling_mean_60m: dict[str, float] | None = None
    rolling_std_60m: dict[str, float] | None = None


class FeedbackPayload(BaseModel):
    prediction_id: str = Field(..., min_length=36, max_length=36)
    actual_failure: bool


def model_row(payload: Telemetry) -> tuple[str | None, dict[str, float], pd.DataFrame]:
    values = payload.model_dump()
    machine_id = values.pop("machine_id", None)
    means = values.pop("rolling_mean_60m", None) or {}
    stds = values.pop("rolling_std_60m", None) or {}
    raw = {feature: float(values[feature]) for feature in RAW_FEATURES}
    engineered = dict(raw)
    for feature in RAW_FEATURES:
        engineered[f"{feature}_mean_60m"] = float(means.get(feature, raw[feature]))
        engineered[f"{feature}_std_60m"] = float(stds.get(feature, 0.0))
    return machine_id, engineered, pd.DataFrame([engineered])[FEATURES]


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(ROOT / "app" / "dashboard.html")


@app.get("/health")
def health() -> dict:
    STORE.ping()
    return {
        "status": "ok",
        "message": "MetroPT-3 compressor condition monitoring API",
        "dataset": ARTIFACT.get("dataset"),
        "monitoring_database": STORE.backend,
        "webhook_configured": NOTIFIER.configured,
    }


@app.get("/demo/telemetry")
def demo_telemetry(
    scenario: str = Query(default="normal", pattern="^(normal|failure)$"),
    step: int = Query(default=0, ge=0),
) -> dict:
    rows = DEMO[DEMO["scenario"] == scenario]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Replay scenario not found")
    row = rows.iloc[step % len(rows)]
    raw = {feature: float(row[feature]) for feature in RAW_FEATURES}
    return {
        "timestamp": row["timestamp"],
        "scenario": scenario,
        "machine_id": "METRO-APU-01",
        **raw,
        "rolling_mean_60m": {feature: float(row[f"{feature}_mean_60m"]) for feature in RAW_FEATURES},
        "rolling_std_60m": {feature: float(row[f"{feature}_std_60m"]) for feature in RAW_FEATURES},
    }


@app.post("/predict")
def predict(payload: Telemetry, background_tasks: BackgroundTasks) -> dict:
    machine_id, values, row = model_row(payload)
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction_id, alert_created = STORE.record_prediction(
        machine_id=machine_id,
        probability=probability,
        threshold=DECISION_THRESHOLD,
        features=values,
    )
    if alert_created:
        background_tasks.add_task(
            NOTIFIER.send,
            {
                "event": "compressor_condition_alert",
                "prediction_id": prediction_id,
                "machine_id": machine_id,
                "risk": round(probability, 4),
                "threshold": DECISION_THRESHOLD,
                "recommended_action": "Acknowledge, inspect pressure circuit, and follow the operator playbook.",
            },
        )
    return {
        "prediction_id": prediction_id,
        "failure_risk": round(probability, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "warning": probability >= DECISION_THRESHOLD,
        "alert_created": alert_created,
        "webhook_configured": NOTIFIER.configured,
        "note": "Condition-monitoring demo trained on the real MetroPT-3 dataset; not a safety system.",
    }


@app.get("/alerts")
def alerts(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    items = STORE.list_alerts(limit)
    return {"count": len(items), "items": items}


@app.get("/monitoring/summary")
def monitoring_summary(hours: int = Query(default=24, ge=1, le=24 * 30)) -> dict:
    return STORE.summary(TRAINING_PROFILE, hours=hours)


@app.post("/feedback")
def feedback(payload: FeedbackPayload) -> dict:
    try:
        STORE.save_feedback(payload.prediction_id, payload.actual_failure)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Prediction not found") from error
    return {"status": "saved", **payload.model_dump()}
