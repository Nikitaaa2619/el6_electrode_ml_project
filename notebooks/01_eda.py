"""EDA for the processed UCI MetroPT-3 minute telemetry."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metropt-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "metropt3_minute.csv.gz"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "eda_summary.md"
TARGET = "air_leak_condition"
SIGNALS = ["tp2_bar", "tp3_bar", "reservoirs_bar", "oil_temperature_c", "motor_current_a"]

FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA, parse_dates=["timestamp"])
missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated(subset=["timestamp"]).sum())

counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(["Нормальное состояние", "Интервал отказа"], counts, color=["#44a08d", "#ff6b57"])
ax.bar_label(bars, labels=[f"{v:,}" for v in counts])
ax.set(title="MetroPT-3: баланс минутных наблюдений", ylabel="Строк")
fig.tight_layout(); fig.savefig(FIG / "01_class_balance.png", dpi=160); plt.close(fig)

sample = pd.concat([
    group.sample(min(len(group), 5000), random_state=42)
    for _, group in df.groupby(TARGET)
])
sample["Состояние"] = sample[TARGET].map({0: "Норма", 1: "Отказ"})
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for feature, ax in zip(SIGNALS, axes.flat):
    sns.boxplot(data=sample, x="Состояние", y=feature, hue="Состояние",
                palette={"Норма": "#44a08d", "Отказ": "#ff6b57"},
                showfliers=False, legend=False, ax=ax)
    ax.set_title(feature); ax.set_xlabel("")
axes.flat[-1].axis("off")
fig.suptitle("Сигналы внутри и вне документированных отказов")
fig.tight_layout(); fig.savefig(FIG / "02_feature_distributions.png", dpi=160); plt.close(fig)

corr = df[SIGNALS + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap="RdBu_r", ax=ax)
ax.set_title("Корреляции основных сигналов")
fig.tight_layout(); fig.savefig(FIG / "03_correlation_heatmap.png", dpi=160); plt.close(fig)

event = df[(df["timestamp"] >= "2020-07-15 12:00") & (df["timestamp"] <= "2020-07-15 21:00")]
fig, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(event["timestamp"], event["reservoirs_bar"], label="Reservoirs, bar", color="#44a08d")
ax1.plot(event["timestamp"], event["tp3_bar"], label="TP3, bar", color="#4277f5")
ax1.fill_between(event["timestamp"], 0, 1, where=event[TARGET].astype(bool),
                 transform=ax1.get_xaxis_transform(), color="#ff6b57", alpha=.2,
                 label="Документированный отказ")
ax1.set(title="Событие 15 июля 2020", ylabel="Давление, bar")
ax1.legend(); fig.tight_layout(); fig.savefig(FIG / "04_failure_timeline.png", dpi=160); plt.close(fig)

split_stats = df.groupby("split")[TARGET].agg(rows="size", failure_share="mean")
summary = f"""# Исследовательский анализ MetroPT-3

## Качество данных

- Минутных строк: {len(df):,}
- Период: {df.timestamp.min()} — {df.timestamp.max()}
- Пропущенных значений: {missing:,}
- Повторяющихся timestamp: {duplicates:,}
- Доля минут внутри отказов: {df[TARGET].mean():.3%}

| Часть | Строк | Доля отказа |
| --- | ---: | ---: |
""" + "\n".join(
    f"| {idx} | {int(row.rows):,} | {row.failure_share:.3%} |"
    for idx, row in split_stats.iterrows()
) + """

![Баланс классов](figures/01_class_balance.png)

## Сигналы

На графиках сравниваются минутные наблюдения внутри и вне интервалов утечки.
Это описательный анализ: различия не доказывают причинность.

![Распределения](figures/02_feature_distributions.png)

![Корреляции](figures/03_correlation_heatmap.png)

## Временное событие

Последнее документированное событие 15 июля полностью оставлено для теста.
Цветная область показывает его интервал.

![Событие](figures/04_failure_timeline.png)

## Вывод

Класс редкий и события различаются между собой. Поэтому случайное перемешивание
строк завысило бы качество: соседние минуты почти одинаковы. В проекте
использовано хронологическое разделение по отдельным событиям, а итог нужно
оценивать также по числу тревожных инцидентов, а не только минут.
"""
REPORT.write_text(summary, encoding="utf-8")
print(summary)
