import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

from src.conf import *
from src.func import annual_risk_return, plot_assets_density

tickers = ["MSFT", "NVDA", "AMZN", "CVX"]

rdata = yf.download(tickers, start="2020-01-01", end="2023-07-31")

data = rdata["Adj Close"].copy()

assert data.isnull().sum().sum() == 0

returns = data.pct_change().dropna()

returns.to_clipboard()
returns.plot(kind="density", figsize=(10, 6))
plot_assets_density(returns)

cov_matrix = returns.cov()

summary = annual_risk_return(returns)

summary["FutureReturns"] = summary.Returns.mul(0.75)

noa = returns.shape[1]
nop = 1_000_000

np.random.seed(1990)
matrix = np.random.random(noa * nop).reshape(nop, noa)
weights = matrix / matrix.sum(axis=1, keepdims=True)

Returns = summary.FutureReturns.dot(weights.T)

Risk = np.sqrt((cov_matrix.dot(weights.T).T * weights).sum(axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(summary.Risk, summary.Returns, c="red", marker="D")
for i in range(noa):
    plt.annotate(summary.index[i], xy=(summary.Risk[i], summary.Returns[i]), size=15)
plt.scatter(Risk, Returns, c=Returns / Risk, marker="o")
plt.show()
