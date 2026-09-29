"""Exploratory analysis for the synthetic electrode telemetry dataset.

Run from the project root:
    python notebooks/01_eda.py

The script writes figures and a short report to reports/.
"""

# %% Imports and paths
from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "el6-matplotlib")
)

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telemetry.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET = "overheat_next_10min"

FEATURE_LABELS = {
    "spindle_temp_c": "Температура шпинделя, °C",
    "bearing_temp_c": "Температура подшипника, °C",
    "temp_rate_c_per_min": "Изменение температуры, °C/мин",
    "tool_wear_pct": "Износ инструмента, %",
    "load_pct": "Нагрузка, %",
    "vibration_rms": "Вибрация RMS",
    "motor_current_a": "Ток двигателя, A",
    "operating_hours_since_service": "Часы после обслуживания",
}


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save one figure with consistent dimensions and resolution."""
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def format_markdown_table(summary: pd.DataFrame) -> str:
    """Build a small Markdown table without an extra dependency."""
    header = "| Признак | Нет риска | Риск | Разница |"
    separator = "| --- | ---: | ---: | ---: |"
    rows = [header, separator]
    for feature, values in summary.iterrows():
        rows.append(
            f"| {FEATURE_LABELS[feature]} | {values['Нет риска']:.2f} | "
            f"{values['Риск']:.2f} | {values['Разница']:+.2f} |"
        )
    return "\n".join(rows)


# %% Load and validate the data
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

required_columns = {"timestamp", "machine_id", TARGET, *FEATURE_LABELS}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

missing_values = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())
overall_risk = float(df[TARGET].mean())
risk_by_machine = (
    df.groupby("machine_id", as_index=False)[TARGET]
    .agg(rows="size", risk_rate="mean")
    .sort_values("machine_id")
)

print(f"Rows: {len(df):,}")
print(f"Missing values: {missing_values}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Overall risk share: {overall_risk:.2%}")
print(risk_by_machine.to_string(index=False))


# %% Figure 1: class balance and machine comparison
class_counts = df[TARGET].value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bars = axes[0].bar(
    ["Нет риска", "Риск"],
    class_counts.values,
    color=["#4C78A8", "#E45756"],
)
axes[0].set_title("Баланс целевого класса")
axes[0].set_ylabel("Количество строк")
axes[0].bar_label(bars, labels=[f"{value:,}" for value in class_counts.values])

sns.barplot(
    data=risk_by_machine,
    x="machine_id",
    y="risk_rate",
    hue="machine_id",
    palette="Blues_d",
    legend=False,
    ax=axes[1],
)
axes[1].set_title("Доля риска по станкам")
axes[1].set_xlabel("Станок")
axes[1].set_ylabel("Доля риска")
axes[1].set_ylim(0, max(0.20, risk_by_machine["risk_rate"].max() * 1.2))
axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f%%", labels=[f"{x:.1%}" for x in container.datavalues])

save_figure(fig, "01_class_balance.png")


# %% Figure 2: key feature distributions by target
key_features = [
    "spindle_temp_c",
    "bearing_temp_c",
    "tool_wear_pct",
    "temp_rate_c_per_min",
]
plot_sample = pd.concat(
    [
        group.sample(min(len(group), 8_000), random_state=42)
        for _, group in df.groupby(TARGET)
    ],
    ignore_index=True,
)
plot_sample["Состояние"] = plot_sample[TARGET].map({0: "Нет риска", 1: "Риск"})

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for feature, ax in zip(key_features, axes.flat):
    sns.boxplot(
        data=plot_sample,
        x="Состояние",
        y=feature,
        hue="Состояние",
        palette={"Нет риска": "#4C78A8", "Риск": "#E45756"},
        showfliers=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(FEATURE_LABELS[feature])
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("Распределения ключевых признаков", fontsize=15, y=1.02)
save_figure(fig, "02_feature_distributions.png")


# %% Figure 3: correlations between numeric signals
corr_features = [
    "spindle_temp_c",
    "bearing_temp_c",
    "tool_wear_pct",
    "operating_hours_since_service",
    "temp_rate_c_per_min",
    "load_pct",
    "vibration_rms",
    "motor_current_a",
    TARGET,
]
corr = df[corr_features].corr()
short_labels = [
    "Шпиндель",
    "Подшипник",
    "Износ",
    "Часы сервиса",
    "Рост температуры",
    "Нагрузка",
    "Вибрация",
    "Ток",
    "Риск",
]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    xticklabels=short_labels,
    yticklabels=short_labels,
    square=True,
    cbar_kws={"label": "Корреляция"},
    ax=ax,
)
ax.set_title("Корреляции ключевых сигналов")
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
save_figure(fig, "03_correlation_heatmap.png")


# %% Figure 4: one risk episode over time
timeline = df[df["machine_id"] == "M-02"].reset_index(drop=True)
risk_starts = timeline.index[timeline[TARGET].diff().fillna(0).eq(1)]
center = int(risk_starts[0]) if len(risk_starts) else len(timeline) // 2
window = timeline.iloc[max(0, center - 60) : min(len(timeline), center + 120)].copy()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(window["timestamp"], window["spindle_temp_c"], label="Шпиндель", linewidth=1.8)
ax.plot(window["timestamp"], window["bearing_temp_c"], label="Подшипник", linewidth=1.5)
ax.fill_between(
    window["timestamp"],
    0,
    1,
    where=window[TARGET].astype(bool),
    color="#E45756",
    alpha=0.18,
    transform=ax.get_xaxis_transform(),
    label="Метка риска",
)
ax.set_title("Пример временного окна M-02")
ax.set_xlabel("Время")
ax.set_ylabel("Температура, °C")
ax.legend(loc="upper left", ncol=3)
save_figure(fig, "04_risk_timeline.png")


# %% Figure 5: risk through the service interval
service_bins = np.arange(0, 225, 25)
service_view = df.assign(
    service_band=pd.cut(
        df["operating_hours_since_service"],
        bins=service_bins,
        right=False,
        include_lowest=True,
    )
)
service_risk = (
    service_view.groupby(["machine_id", "service_band"], observed=True)[TARGET]
    .mean()
    .rename("risk_rate")
    .reset_index()
)
service_risk["service_hours"] = service_risk["service_band"].map(
    lambda interval: float(interval.mid)
)

fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(
    data=service_risk,
    x="service_hours",
    y="risk_rate",
    hue="machine_id",
    marker="o",
    ax=ax,
)
ax.set_title("Риск по мере накопления часов после обслуживания")
ax.set_xlabel("Часы после обслуживания")
ax.set_ylabel("Доля риска")
ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
ax.legend(title="Станок")
save_figure(fig, "05_risk_by_service_hours.png")


# %% Build the Markdown summary
summary_features = [
    "spindle_temp_c",
    "bearing_temp_c",
    "temp_rate_c_per_min",
    "tool_wear_pct",
    "load_pct",
    "vibration_rms",
    "motor_current_a",
    "operating_hours_since_service",
]
class_means = df.groupby(TARGET)[summary_features].mean().T
class_means.columns = ["Нет риска", "Риск"]
class_means["Разница"] = class_means["Риск"] - class_means["Нет риска"]

machine_lines = "\n".join(
    f"- `{row.machine_id}`: {row.risk_rate:.2%} риска ({int(row.rows):,} строк)"
    for row in risk_by_machine.itertuples(index=False)
)

report = f"""# Исследовательский анализ синтетической телеметрии

## Качество и состав данных

- Строк: {len(df):,}
- Колонок: {df.shape[1]}
- Период: {df['timestamp'].min()} — {df['timestamp'].max()}
- Пропущенных значений: {missing_values}
- Полных дубликатов: {duplicate_rows}
- Общая доля риска: {overall_risk:.2%}

{machine_lines}

![Баланс класса и станки](figures/01_class_balance.png)

## Различия между классами

{format_markdown_table(class_means)}

![Распределения признаков](figures/02_feature_distributions.png)

Риск связан прежде всего с температурой шпинделя и подшипника, износом,
часами после обслуживания и положительной скоростью изменения температуры.
Это ожидаемо: генератор использует эти сигналы при создании искусственной
метки. Корреляция показывает совместное изменение, а не причинность.

![Корреляции](figures/03_correlation_heatmap.png)

## Временная динамика и обслуживание

Красная область на временном графике показывает минуты, для которых генератор
поставил метку риска в следующие 10 минут.

![Пример временного окна](figures/04_risk_timeline.png)

Риск растёт ближе к концу искусственного 200-часового интервала обслуживания.
Различия между станками заданы профилями генератора: у `M-02` выше базовая
нагрузка и температура, у `M-03` — ниже.

![Риск и обслуживание](figures/05_risk_by_service_hours.png)

## Выводы для модели

1. Accuracy нельзя оценивать отдельно: около 90% строк относятся к классу без
   риска, поэтому простая стратегия «всегда нет риска» уже получает высокую
   accuracy и при этом пропускает все предупреждения.
2. Температуры, износ и часы после обслуживания сильно связаны друг с другом.
   Важность признаков Random Forest не следует трактовать как причинность.
3. Порог предупреждения нужно выбирать на отдельной валидационной части,
   сравнивая пропущенные риски с количеством ложных тревог.
4. Все закономерности созданы формулами генератора. Для реального применения
   потребуются реальные данные, проверка разметки и анализ с инженерами.
"""

(REPORT_DIR / "eda_summary.md").write_text(report, encoding="utf-8")
print(f"Saved report to {REPORT_DIR / 'eda_summary.md'}")
print(f"Saved figures to {FIGURE_DIR}")
