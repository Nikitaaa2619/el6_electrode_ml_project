from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "metropt3_minute.csv.gz"
DEMO_OUTPUT = ROOT / "data" / "metropt3_demo.csv"

SOURCE_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

RENAME = {
    "TP2": "tp2_bar",
    "TP3": "tp3_bar",
    "H1": "h1_bar",
    "DV_pressure": "dv_pressure_bar",
    "Reservoirs": "reservoirs_bar",
    "Oil_temperature": "oil_temperature_c",
    "Motor_current": "motor_current_a",
    "COMP": "comp",
    "DV_eletric": "dv_electric",
    "Towers": "towers",
    "MPG": "mpg",
    "LPS": "lps",
    "Pressure_switch": "pressure_switch",
    "Oil_level": "oil_level",
    "Caudal_impulses": "caudal_impulses",
}

# Failure intervals published with MetroPT-3 by the equipment operator.
FAILURES = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def prepare(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        source,
        usecols=["timestamp", *SOURCE_COLUMNS],
        parse_dates=["timestamp"],
        dtype={column: "float32" for column in SOURCE_COLUMNS},
    )
    raw["timestamp"] = raw["timestamp"].dt.floor("min")
    minute = raw.groupby("timestamp", as_index=False)[SOURCE_COLUMNS].mean()
    minute = minute.rename(columns=RENAME).sort_values("timestamp")
    features = list(RENAME.values())
    rolling = minute.set_index("timestamp")[features].rolling("60min", min_periods=1)
    rolling_mean = rolling.mean().add_suffix("_mean_60m")
    rolling_std = rolling.std().fillna(0).add_suffix("_std_60m")
    minute = minute.set_index("timestamp").join([rolling_mean, rolling_std]).reset_index()

    minute["air_leak_condition"] = 0
    for start_text, end_text in FAILURES:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        in_failure = (minute["timestamp"] >= start) & (minute["timestamp"] <= end)
        minute.loc[in_failure, "air_leak_condition"] = 1
    minute["split"] = "train"
    minute.loc[minute["timestamp"] >= "2020-06-01", "split"] = "validation"
    minute.loc[minute["timestamp"] >= "2020-06-20", "split"] = "test"
    return minute


def save_demo_rows(data: pd.DataFrame) -> None:
    normal = data[
        (data["timestamp"] >= "2020-07-10 08:00")
        & (data["timestamp"] < "2020-07-10 10:00")
    ].head(90).copy()
    normal["scenario"] = "normal"

    failure = data[
        (data["timestamp"] >= "2020-07-15 14:30:00")
        & (data["timestamp"] < "2020-07-15 16:00:00")
    ].copy()
    failure["scenario"] = "failure"
    demo = pd.concat([normal, failure], ignore_index=True)
    demo.to_csv(DEMO_OUTPUT, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the official MetroPT-3 CSV into a reproducible minute dataset."
    )
    parser.add_argument("source", type=Path, help="Path to MetroPT3(AirCompressor).csv")
    args = parser.parse_args()

    data = prepare(args.source)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False, compression="gzip")
    save_demo_rows(data)

    print(f"Saved {len(data):,} minute rows to {OUTPUT}")
    print(data.groupby("split")["air_leak_condition"].agg(["count", "sum", "mean"]))
    print(f"Saved replay rows to {DEMO_OUTPUT}")


if __name__ == "__main__":
    main()
