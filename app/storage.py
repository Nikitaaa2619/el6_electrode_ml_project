from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    machine_id: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)
    probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    warning: Mapped[bool] = mapped_column(Boolean, index=True)
    features: Mapped[dict[str, float]] = mapped_column(JSON)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    machine_id: Mapped[str] = mapped_column(String(50), index=True)
    probability: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(String(300))


class Feedback(Base):
    __tablename__ = "feedback"

    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), primary_key=True)
    actual_overheat: Mapped[bool] = mapped_column(Boolean)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _database_url(root: Path) -> str:
    configured = os.getenv("DATABASE_URL")
    if configured:
        # Render currently supplies postgresql://. Explicitly select the psycopg 3 driver.
        if configured.startswith("postgresql://"):
            return configured.replace("postgresql://", "postgresql+psycopg://", 1)
        if configured.startswith("postgres://"):
            return configured.replace("postgres://", "postgresql+psycopg://", 1)
        return configured

    path = Path(os.getenv("STATE_DB_PATH", root / "storage" / "monitoring.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


class MonitoringStore:
    def __init__(self, root: Path) -> None:
        url = _database_url(root)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self.backend = "postgresql" if url.startswith("postgresql") else "sqlite"
        self.alert_cooldown = timedelta(
            seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "900"))
        )

    def ping(self) -> None:
        with self.sessions() as session:
            session.execute(select(1))

    def record_prediction(
        self,
        *,
        machine_id: str | None,
        probability: float,
        threshold: float,
        features: dict[str, float],
    ) -> tuple[str, bool]:
        now = datetime.now(timezone.utc)
        prediction_id = str(uuid4())
        warning = probability >= threshold
        alert_created = False

        with self.sessions() as session:
            session.add(
                Prediction(
                    id=prediction_id,
                    created_at=now,
                    machine_id=machine_id,
                    probability=probability,
                    threshold=threshold,
                    warning=warning,
                    features=features,
                )
            )

            if warning and machine_id:
                last_alert = session.scalars(
                    select(Alert)
                    .where(Alert.machine_id == machine_id)
                    .order_by(Alert.created_at.desc())
                    .limit(1)
                ).first()
                if not last_alert or self._as_utc(last_alert.created_at) < now - self.alert_cooldown:
                    session.add(
                        Alert(
                            prediction_id=prediction_id,
                            created_at=now,
                            machine_id=machine_id,
                            probability=probability,
                            message="Обнаружено состояние, похожее на документированный режим утечки воздуха.",
                        )
                    )
                    alert_created = True
            session.commit()
        return prediction_id, alert_created

    def list_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.sessions() as session:
            alerts = session.scalars(
                select(Alert).order_by(Alert.created_at.desc()).limit(limit)
            ).all()
            feedback = {
                item.prediction_id: item.actual_overheat
                for item in session.scalars(
                    select(Feedback).where(
                        Feedback.prediction_id.in_([a.prediction_id for a in alerts])
                    )
                ).all()
            } if alerts else {}

        return [
            {
                "id": alert.id,
                "prediction_id": alert.prediction_id,
                "created_at": self._as_utc(alert.created_at).isoformat(),
                "machine_id": alert.machine_id,
                "probability": round(alert.probability, 4),
                "message": alert.message,
                "actual_failure": feedback.get(alert.prediction_id),
            }
            for alert in alerts
        ]

    def save_feedback(self, prediction_id: str, actual_failure: bool) -> None:
        with self.sessions() as session:
            if session.get(Prediction, prediction_id) is None:
                raise KeyError(prediction_id)
            item = session.get(Feedback, prediction_id)
            if item:
                item.actual_overheat = actual_failure
                item.recorded_at = datetime.now(timezone.utc)
            else:
                session.add(
                    Feedback(
                        prediction_id=prediction_id,
                        actual_overheat=actual_failure,
                        recorded_at=datetime.now(timezone.utc),
                    )
                )
            session.commit()

    def summary(
        self,
        training_profile: dict[str, dict[str, float]],
        *,
        hours: int = 24,
    ) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.sessions() as session:
            predictions = session.scalars(
                select(Prediction)
                .where(Prediction.created_at >= since)
                .order_by(Prediction.created_at.desc())
                .limit(1000)
            ).all()
            labeled = session.execute(
                select(Prediction.warning, Feedback.actual_overheat)
                .join(Feedback, Feedback.prediction_id == Prediction.id)
                .order_by(Feedback.recorded_at.desc())
                .limit(1000)
            ).all()

        probabilities = [item.probability for item in predictions]
        drift = self._feature_drift(predictions, training_profile)
        quality = self._quality_metrics(labeled)
        return {
            "window_hours": hours,
            "database_backend": self.backend,
            "request_count": len(predictions),
            "warning_count": sum(item.warning for item in predictions),
            "warning_rate": round(float(np.mean([item.warning for item in predictions])), 4)
            if predictions
            else 0.0,
            "average_probability": round(float(np.mean(probabilities)), 4)
            if probabilities
            else 0.0,
            "p95_probability": round(float(np.percentile(probabilities, 95)), 4)
            if probabilities
            else 0.0,
            "drift": drift,
            "quality": quality,
        }

    @staticmethod
    def _feature_drift(
        predictions: list[Prediction],
        training_profile: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        if len(predictions) < 20:
            return {"status": "insufficient_data", "sample_size": len(predictions), "features": []}

        feature_rows = [item.features for item in predictions]
        results = []
        for feature, reference in training_profile.items():
            values = [float(row[feature]) for row in feature_rows if feature in row]
            std = max(float(reference.get("std", 0.0)), 1e-9)
            if not values:
                continue
            z_shift = abs(float(np.mean(values)) - float(reference["mean"])) / std
            if z_shift >= 1.0:
                results.append(
                    {
                        "feature": feature,
                        "z_shift": round(z_shift, 2),
                        "current_mean": round(float(np.mean(values)), 3),
                        "training_mean": round(float(reference["mean"]), 3),
                    }
                )
        results.sort(key=lambda item: item["z_shift"], reverse=True)
        return {
            "status": "drift_detected" if results else "stable",
            "sample_size": len(predictions),
            "features": results[:5],
        }

    @staticmethod
    def _quality_metrics(labeled: list[tuple[bool, bool]]) -> dict[str, Any]:
        if not labeled:
            return {
                "labeled_count": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }

        predicted = np.array([int(row[0]) for row in labeled])
        actual = np.array([int(row[1]) for row in labeled])
        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        fn = int(((predicted == 0) & (actual == 1)).sum())
        tn = int(((predicted == 0) & (actual == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "labeled_count": len(labeled),
            "accuracy": round((tp + tn) / len(labeled), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall
            else 0.0,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
