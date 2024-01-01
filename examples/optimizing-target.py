import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ta
import yfinance as yf


def scenarios(temp, n=30, lower=-0.6, upper=1.2):
    # n = 200
    # lower = -3.9
    # upper = 3.9
    # temp = df1.copy()
    temp = temp[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            # "trend_macd",
            # "trend_macd_signal",
            # "trend_macd_diff",
            # "momentum_rsi",
            # "volume_vwap",
        ]
    ].copy()
    Cumulated_name = []
    Cumulated_boundary = []
    for i in range(1, n + 1):
        # i = 1
        # print(i)
        first_name = "DiffClose" + str(i)
        second_name = "DiffMin" + str(i)
        temp[first_name] = (
            temp["close"].diff(i).shift(-i).fillna(0) / temp["close"] * 100
        )
        temp[second_name] = np.where(
            (temp[first_name] < lower) | (temp[first_name] > upper), i, n
        )
        Cumulated_name.append(first_name)
        Cumulated_boundary.append(second_name)
    temp["Period"] = temp[Cumulated_boundary].min(axis=1)
    temp["Outcome"] = temp.apply(
        lambda x: x["DiffClose" + str(x["Period"]).split(".")[0]], axis=1
    )
    outcome = temp[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "Period",
            "Outcome",
            # "trend_macd",
            # "trend_macd_signal",
            # "trend_macd_diff",
            # "momentum_rsi",
            # "volume_vwap",
        ]
    ]
    return outcome


def plotting(indicator):
    sns.set(font_scale=1.3)
    df = indicator.groupby("Outcome").size().reset_index()
    df.columns = ["Outcome", "Cases %"]
    df["Total_Cases"] = df["Cases %"].sum()
    df["Cases %"] = df["Cases %"] / df["Total_Cases"] * 100
    plt.figure(figsize=(14, 7))
    g = sns.barplot(data=df, x="Outcome", y="Cases %", palette="Pastel1")
    plt.title("Combination Title")
    for index, row in df.iterrows():
        g.text(
            row.name,
            row["Cases %"],
            str(round(row["Cases %"], 2)) + "%",
            fontsize=20,
            color="black",
            ha="center",
        )
    plt.show()


data = yf.download("TSLA", start="2023-01-01", end="2023-12-28")

df = data.copy()
df.columns = [x.lower() for x in df.columns]

df1 = ta.add_all_ta_features(
    df, open="open", high="high", low="low", close="close", volume="volume"
)

matrix = list(product(np.arange(3, 42, 3) / 10, np.arange(-3, -42, -3) / 10))

table = []
for i in matrix:  # matrix[len(matrix) - 1 : len(matrix)]:
    # print(i)
    final = scenarios(df1, n=200, upper=i[0], lower=i[1])
    # collect metadata
    count = np.sign(final["Outcome"]).value_counts().sort_index()
    distr = np.sign(final["Outcome"]).value_counts(normalize=True).sort_index()
    table.append(
        [
            i[0],
            i[1],
            distr[1],
            distr[0],
            distr[-1],
            count[1],
            count[0],
            count[-1],
        ]
    )
    # adjust columns
    final["Outcome"] = np.where(final["Outcome"] > 0, str(i[0]) + "%", str(i[1]) + "%")
    final["Outcome"] = np.where(final["Period"] == 200, "No Outcome", final["Outcome"])
    # plotting(final)
    # final.to_csv("Model_data" + str(i) + ".csv")

tdf = pd.DataFrame(
    table,
    columns=[
        "Upper",
        "Lower",
        "Up",
        "No Outcome",
        "Down",
        "Up Cases",
        "No Outcome Cases",
        "Down Cases",
    ],
)

FEES = 0.15
SUFFIX = ""  # " Cases"
tdf["E_UP"] = (tdf["Upper"] - FEES) * tdf["Up" + SUFFIX]
tdf["E_DOWN"] = (tdf["Lower"] - FEES) * tdf["Down" + SUFFIX]
tdf["E_STALE"] = (0 - FEES) * tdf["No Outcome" + SUFFIX]
tdf["E_TOTAL"] = tdf["E_UP"] + tdf["E_DOWN"] + tdf["E_STALE"]
tdf["E_RANK"] = tdf["E_TOTAL"].rank(ascending=False)

# Create a EXPECTED VALUE Heatmap
pivot_table = tdf.pivot_table(index="Lower", columns="Upper", values="E_TOTAL")
# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    pivot_table, cmap="RdYlGn", center=0, annot=True, fmt=".1f", annot_kws={"size": 9}
)
plt.title("Expected Value")

# # Export to Excel
# tdf.to_clipboard(index=False)
# pivot_table.to_clipboard()

# Create a EXPECTED VALUE RANK Heatmap
pivot_table_rank = tdf.pivot_table(index="Lower", columns="Upper", values="E_RANK")
# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    pivot_table_rank,
    cmap="RdYlGn_r",
    center=0,
    annot=True,
    fmt=".0f",
    annot_kws={"size": 9},
)
plt.title("Expected Value Rank")
