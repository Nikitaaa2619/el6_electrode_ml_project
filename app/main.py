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
DECISION_THRESHOLD = ARTIFACT.get("threshold", 0.5)
TRAINING_PROFILE = ARTIFACT.get("training_profile", {})
STORE = MonitoringStore(ROOT)
NOTIFIER = AlertNotifier()

app = FastAPI(title="Industrial Overheat Prediction API")


class Telemetry(BaseModel):
    machine_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    electrode_diameter_mm: float = Field(..., gt=0)
    ambient_temp_c: float
    spindle_rpm: float = Field(..., ge=0)
    load_pct: float = Field(..., ge=0, le=100)
    motor_current_a: float = Field(..., ge=0)
    vibration_rms: float = Field(..., ge=0)
    coolant_temp_c: float
    pressure_bar: float = Field(..., ge=0)
    operating_hours_since_service: float = Field(..., ge=0)
    tool_wear_pct: float = Field(..., ge=0, le=100)
    bearing_temp_c: float
    spindle_temp_c: float
    temp_rate_c_per_min: float


class FeedbackPayload(BaseModel):
    prediction_id: str = Field(..., min_length=36, max_length=36)
    actual_overheat: bool


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(ROOT / "app" / "dashboard.html")


@app.get("/health")
def health() -> dict:
    STORE.ping()
    return {
        "status": "ok",
        "message": "Industrial overheat prediction API",
        "monitoring_database": STORE.backend,
        "webhook_configured": NOTIFIER.configured,
    }


@app.post("/predict")
def predict(payload: Telemetry, background_tasks: BackgroundTasks) -> dict:
    values = payload.model_dump()
    machine_id = values.pop("machine_id", None)
    row = pd.DataFrame([values])[FEATURES]
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction_id, alert_created = STORE.record_prediction(
        machine_id=machine_id,
        probability=probability,
        threshold=DECISION_THRESHOLD,
        features={key: float(values[key]) for key in FEATURES},
    )
    if alert_created:
        background_tasks.add_task(
            NOTIFIER.send,
            {
                "event": "electrode_machine_overheat_alert",
                "prediction_id": prediction_id,
                "machine_id": machine_id,
                "risk": round(probability, 4),
                "threshold": DECISION_THRESHOLD,
                "recommended_action": "Acknowledge and follow the approved overheat response playbook.",
            },
        )
    return {
        "prediction_id": prediction_id,
        "overheat_probability": round(probability, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "warning": probability >= DECISION_THRESHOLD,
        "alert_created": alert_created,
        "webhook_configured": NOTIFIER.configured,
        "note": "Educational simulation only; not a real machine-control system.",
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
        STORE.save_feedback(payload.prediction_id, payload.actual_overheat)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Prediction not found") from error
    return {"status": "saved", **payload.model_dump()}
