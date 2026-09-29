from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = joblib.load(ROOT / "models" / "model.joblib")
MODEL = ARTIFACT["model"]
FEATURES = ARTIFACT["features"]
DECISION_THRESHOLD = ARTIFACT.get("threshold", 0.5)

app = FastAPI(title="Industrial Overheat Prediction API")


class Telemetry(BaseModel):
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


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(ROOT / "app" / "dashboard.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "Industrial overheat prediction API"}


@app.post("/predict")
def predict(payload: Telemetry) -> dict:
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    probability = float(MODEL.predict_proba(row)[0, 1])
    return {
        "overheat_probability": round(probability, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "warning": probability >= DECISION_THRESHOLD,
        "note": "Educational simulation only; not a real machine-control system.",
    }
