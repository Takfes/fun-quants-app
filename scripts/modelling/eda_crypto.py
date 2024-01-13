import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from quanttak.data import fetch_ohlcv_data

"""
# ==============================================================
# Define Process
# ==============================================================
"""

timeframe = "1d"
symbolsinput = [
    "ADA",
    "ETH",
    "BTC",
    "VET",
    "SOL",
    "XRP",
    "LINK",
    "AVAX",
    "MATIC",
    "DOT",
]
symbols = sorted([f"{x}/USDT" for x in symbolsinput])

# Download Data
data = []
for symbol in tqdm(symbols):
    temp = fetch_ohlcv_data(symbol, timeframe)
    temp.insert(0, "symbol", symbol)
    data.append(temp.set_index("timestamp"))

# Calculate Correlations
for i, d in enumerate(data):
    if i == 0:
        prices = pd.DataFrame(index=d.index)
        returns = pd.DataFrame(index=d.index)
    s = d.symbol.unique()[0]
    prices = prices.join(d.close.rename(s))
    returns = returns.join(d.close.pct_change().rename(s))

correlation_price = prices.corr()
correlation_return = returns.dropna().corr()

# Plot Correlation Heatmaps
fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
sns.heatmap(correlation_price, annot=True, cmap="coolwarm", ax=axs[0])
axs[0].set_title("Price Correlation")
axs[0].set_xlabel("Symbols")
axs[0].set_ylabel("Symbols")
sns.heatmap(correlation_return, annot=True, cmap="coolwarm", ax=axs[1])
axs[1].set_title("Returns Correlation")
axs[1].set_xlabel("Symbols")
axs[1].set_ylabel("Symbols")
plt.tight_layout()
plt.show()


# Assuming you have a series called 'data'
rd = returns.dropna()
desc = rd.describe().T
desc.insert(desc.shape[1], "skew", rd.skew())
desc.insert(desc.shape[1], "kurtosis", rd.kurtosis())
desc.insert(desc.columns.tolist().index("min") + 1, "5%", rd.quantile(0.05))
desc.insert(desc.columns.tolist().index("min") + 1, "1%", rd.quantile(0.01))
desc.insert(desc.columns.tolist().index("max") - 1, "95%", rd.quantile(0.95))
desc.insert(desc.columns.tolist().index("max") - 1, "99%", rd.quantile(0.99))
desc.drop("count", axis=1, inplace=True)
desc.sort_values("1%", ascending=True)

# # Create a violin plot from returns dataframe
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.violinplot(data=rd, ax=ax, orient="h", palette="Set2")
# ax.set_title("Returns Distribution")
# ax.set_xlabel("Symbols")
# ax.set_ylabel("Returns")
# plt.show()
