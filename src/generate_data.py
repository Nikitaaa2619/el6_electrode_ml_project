from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
OUT = Path(__file__).resolve().parents[1] / "data" / "telemetry.csv"
FORECAST_HORIZON_MINUTES = 10
SERVICE_INTERVAL_HOURS = 200.0
MACHINE_PROFILES = {
    "M-01": {"load_offset": 0.0, "temp_offset": 0.0},
    "M-02": {"load_offset": 3.0, "temp_offset": 1.5},
    "M-03": {"load_offset": -2.0, "temp_offset": -1.0},
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_machine(
    rng: np.random.Generator,
    machine_id: str,
    n: int,
    load_offset: float,
    temp_offset: float,
) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    electrode_diameter = rng.choice([200, 250, 300, 350, 400], size=n)

    # Synthetic production telemetry. These ranges are assumptions for training only.
    ambient = rng.normal(22, 3.5, n).clip(10, 35)
    rpm = rng.normal(1800, 280, n).clip(900, 2600)
    load = rng.normal(58 + load_offset, 15, n).clip(15, 95)
    current = (12 + load * 0.11 + rng.normal(0, 1.4, n)).clip(7, 30)
    vibration = (1.2 + load * 0.018 + rng.normal(0, 0.25, n)).clip(0.3, 4.5)
    coolant_temp = (ambient + 10 + load * 0.08 + rng.normal(0, 1.2, n)).clip(15, 45)
    pressure = (3.1 - load * 0.006 + rng.normal(0, 0.15, n)).clip(1.8, 3.8)
    service_start = rng.uniform(0, SERVICE_INTERVAL_HOURS)
    service_hours = (service_start + np.arange(n) / 60.0) % SERVICE_INTERVAL_HOURS
    tool_wear = (
        service_hours / SERVICE_INTERVAL_HOURS * 100 + rng.normal(0, 5, n)
    ).clip(0, 100)

    # Latent heat/load dynamics.
    base_temp = 38 + temp_offset + 0.20 * load + 0.004 * rpm + 0.08 * ambient
    bearing_temp = base_temp + 0.75 * vibration + 0.12 * tool_wear + rng.normal(0, 2.0, n)
    spindle_temp = bearing_temp + 3 + 0.10 * current + rng.normal(0, 1.2, n)

    # Inject rare degradation episodes (synthetic).
    episode_starts = rng.choice(np.arange(20, n - 40), size=40, replace=False)
    episode_strength = rng.uniform(8, 22, size=len(episode_starts))
    for start, strength in zip(episode_starts, episode_strength):
        length = int(rng.integers(8, 30))
        end = min(n, start + length)
        ramp = np.linspace(0.2, 1.0, end - start)
        spindle_temp[start:end] += strength * ramp
        bearing_temp[start:end] += strength * 0.8 * ramp
        vibration[start:end] += 0.8 * ramp
        current[start:end] += 2.5 * ramp
        load[start:end] += 10 * ramp

    load = load.clip(0, 100)
    current = current.clip(0, 30)
    vibration = vibration.clip(0, 4.5)

    # Estimate current temperature growth; lagged values are used to avoid future leakage.
    temp_series = pd.Series(spindle_temp)
    temp_rate = temp_series.diff().rolling(5).mean().fillna(0).to_numpy()

    # Label = future overheating event in next 10 minutes, based on synthetic temperature trajectory.
    future_max = (
        pd.Series(spindle_temp)
        .shift(-1)
        .rolling(FORECAST_HORIZON_MINUTES, min_periods=FORECAST_HORIZON_MINUTES)
        .max()
        .shift(-(FORECAST_HORIZON_MINUTES - 1))
        .to_numpy()
    )
    # A second signal makes the task less trivial: rapid heating can trigger risk even before the hard threshold.
    risk_score = (
        (future_max - 82) / 4.5
        + 0.45 * temp_rate
        + 0.40 * (vibration - 2.8)
        + 0.02 * (tool_wear - 70)
    )
    probability = sigmoid(risk_score)
    label = (probability > 0.70).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_id,
        "electrode_diameter_mm": electrode_diameter,
        "ambient_temp_c": np.round(ambient, 3),
        "spindle_rpm": np.round(rpm, 3),
        "load_pct": np.round(load, 3),
        "motor_current_a": np.round(current, 3),
        "vibration_rms": np.round(vibration, 4),
        "coolant_temp_c": np.round(coolant_temp, 3),
        "pressure_bar": np.round(pressure, 3),
        "operating_hours_since_service": np.round(service_hours, 2),
        "tool_wear_pct": np.round(tool_wear, 2),
        "bearing_temp_c": np.round(bearing_temp, 3),
        "spindle_temp_c": np.round(spindle_temp, 3),
        "temp_rate_c_per_min": np.round(temp_rate, 4),
        "overheat_next_10min": label,
    })

    # The last rows of each machine cannot have a full future window.
    return df.iloc[:-FORECAST_HORIZON_MINUTES].copy()


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    rows_per_machine = 20_000
    frames = [
        generate_machine(rng, machine_id, rows_per_machine, **profile)
        for machine_id, profile in MACHINE_PROFILES.items()
    ]
    df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "machine_id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df):,} rows to {OUT}")
    print(df["overheat_next_10min"].value_counts(normalize=True).rename("share"))


if __name__ == "__main__":
    main()
