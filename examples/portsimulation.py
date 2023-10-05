import time
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import yfinance as yf
from matplotlib import pyplot as plt

from src.conf import *

sns.set_style("darkgrid")

# ===============================================
# Define functions
# ===============================================


# def annual_risk_return(data: pd.DataFrame, risk_free_rate: float = 0.017) -> pd.DataFrame:
#     summary = data.agg(["mean", "std"]).T
#     summary.columns = ["Returns", "Risk"]
#     summary.Returns = summary.Returns * 252
#     summary.Risk = summary.Risk * np.sqrt(252)
#     summary["Sharpe"] = (summary.Returns - risk_free_rate) / summary.Risk
#     return summary


def annual_risk_return(
    data: pd.DataFrame, risk_free_rate: float = 0.017
) -> pd.DataFrame:
    # Convert pandas DataFrame to polars DataFrame
    pl_data = pl.from_pandas(data)
    # Calculate mean and standard deviation using polars
    returns = pl_data.select(
        [(pl.col(column).mean().alias(column)) for column in pl_data.columns]
    )
    risk = pl_data.select(
        [(pl.col(column).std().alias(column)) for column in pl_data.columns]
    )
    # Convert the results back to pandas DataFrames
    pd_returns = returns.to_pandas().T
    pd_risk = risk.to_pandas().T
    # Create the summary pandas DataFrame
    summary = pd.DataFrame(
        {
            "Returns": pd_returns.sum(axis=1) * 252,
            "Risk": pd_risk.sum(axis=1) * np.sqrt(252),
        }
    )
    summary["Sharpe"] = (summary.Returns - risk_free_rate) / summary.Risk
    return summary


def find_below_threshold_missingness(
    data: pd.DataFrame, threshold: float = 0.0
) -> List:
    return (
        (data.isnull().sum() / data.shape[0])
        .loc[lambda x: x <= threshold]
        .index.tolist()
    )


def plot_portfolios(
    assets_data: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    annotate: bool = False,
    size: int = 15,
):
    # calculate max sharpe ratio portfolio
    max_sharpe_idx = portfolio_data.Sharpe.idxmax()
    max_sharpe = portfolio_data.loc[max_sharpe_idx]
    plt.figure(figsize=(15, 8))
    # plot generated portfolios
    plt.scatter(
        x=portfolio_data.loc[:, "Risk"],
        y=portfolio_data.loc[:, "Returns"],
        c=portfolio_data.loc[:, "Sharpe"],
        cmap="coolwarm",
        s=15,
        vmin=0.5,
        vmax=1.00,
        alpha=0.8,
    )
    plt.colorbar()
    # plot original assets used to generate portfolios
    plt.scatter(
        x=assets_data.loc[:, "Risk"],
        y=assets_data.loc[:, "Returns"],
        c=assets_data.loc[:, "Sharpe"],
        cmap="coolwarm",
        s=60,
        vmin=0.5,
        vmax=1.00,
        alpha=0.8,
        marker="D",
    )
    if annotate:
        for i in assets_data.index:
            plt.annotate(
                i,
                xy=(
                    assets_data.loc[assets_data.index == i, "Risk"].squeeze(),
                    assets_data.loc[assets_data.index == i, "Returns"].squeeze(),
                ),
                size=size,
            )
    # plot max sharpe ratio portfolio
    plt.scatter(
        x=max_sharpe["Risk"],
        y=max_sharpe["Returns"],
        c="black",
        s=300,
        marker="*",
    )
    plt.xlabel("Ann Risk (std)", fontsize=15)
    plt.ylabel("Ann Returns", fontsize=15)
    plt.title("Risk/Return/Sharpe Ratio", fontsize=20)
    plt.show()


# ===============================================
# Get data
# ===============================================

START_DATE = "2022-01-01"
END_DATE = "2023-07-31"
THRESHOLD = 0.05
PORTFOLIO = SP_YIELD_KING
TICKERS = list(PORTFOLIO.keys())

# Download data
rdata = yf.download(TICKERS, start=START_DATE, end=END_DATE, actions=True)

# Keep only the Adj Close column
data = rdata["Adj Close"].copy()

# ===============================================
# Clean data
# ===============================================

# observe data presence (opposite of missingness)
round(1 - data.isnull().sum() / data.shape[0], 2).sort_values(ascending=True).head(10)

# Track symbols with acceptable missingness
non_missing = find_below_threshold_missingness(data, threshold=THRESHOLD)

# Keep only symbols with acceptable missingness
datac = data[non_missing].copy()

# find returns and drop nas
returns = datac.pct_change().dropna()
# returns = np.log(datac / datac.shift(1)).dropna()

assert datac.shape[1] == len(
    non_missing
), "Number of symbols with acceptable missingness does not match"
assert (
    datac.shape[0] - int(datac.shape[0] * THRESHOLD) <= returns.shape[0]
), "Number of trading days differs significantly"

print(f"> Data start date: {datac.index[0]}")
print(f"> Data end date: {datac.index[-1]}")
print(f"* Number of symbols intended to fetch {len(TICKERS)}")
print(
    f"* Number of symbols with acceptable missingness ({THRESHOLD:.2%}) : {len(non_missing)}"
)
print(
    f"* Number of symbols dropped due to missingness {len(TICKERS) - len(non_missing)}"
)
print(f"* Symbols dropped due to missingness {set(TICKERS) - set(non_missing)}")
print(f"# Number of days in data: {datac.shape[0]}")
print(f"# Number of days with returns: {returns.shape[0]}")
print(
    f"# Number of days dropped due to missingness {datac.shape[0] - returns.shape[0]}"
)

# ===============================================
# Create Random Portfolios
# ===============================================

# Set number of portfolios
NOP = 1_000_000

# Consituents Summary
assets_summary = annual_risk_return(returns)

# Set number of assets
NOA = assets_summary.shape[0]
returns_to_use = returns.copy()

# Set weights
np.random.seed(123)
matrix = np.random.random(NOP * NOA).reshape(NOP, NOA)
weights = matrix / matrix.sum(axis=1).reshape(-1, 1)
all(weights.sum(axis=1, keepdims=True)) == 1

# Generate random portfolios
port_ret = returns_to_use.dot(weights.T)

# Calculate Risk and Return for the random portfolios
start = time.perf_counter()
portfolios_summary = annual_risk_return(port_ret)
end = time.perf_counter()
print(f"Time taken: {end - start:.2f} seconds")

# Plot Random Portfolios
plot_portfolios(assets_summary, portfolios_summary, annotate=True)

# find max sharpe ratio
max_sharpe_idx = portfolios_summary.Sharpe.idxmax()
max_sharpe_weights = weights[int(max_sharpe_idx), :]
max_sharpe = portfolios_summary.loc[max_sharpe_idx]

# find total returns
weighted_returns = returns_to_use.dot(max_sharpe_weights)
simple_returns = weighted_returns.sum()
compounded_returns = (weighted_returns + 1).prod() - 1
analysis_weights = dict(zip(TICKERS, np.round(max_sharpe_weights, 4)))

print(f"Simple Returns: {simple_returns:.2%}")
print(f"Compounded Returns: {compounded_returns:.2%}")
print(f"Sharpe Ratio Portfolio:\n{max_sharpe}")
print(f"Weights:{analysis_weights}")


# ===============================================
# Optimization
# ===============================================

# calculate returns and drop nas
returns = datacand.pct_change().dropna()

# Calculate expected returns and the covariance matrix
expected_returns = returns.mean()
covariance_matrix = returns.cov()

# Set up the optimization problem
num_assets = len(ticker_candidates)
weights = np.array([1.0 / num_assets] * num_assets)  # Starting with equal weights

# Constraints
constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1,
}  # Weights sum to 1

# Boundaries for weights (can be adjusted depending on requirements)
bounds = tuple((0.05, 1) for asset in range(num_assets))


# Objective Function (We want to maximize Sharpe Ratio, so we minimize the negative Sharpe Ratio)
def objective(weights):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return (
        -portfolio_return / portfolio_stddev
    )  # Negative Sharpe Ratio (assuming risk-free rate is 0 for simplicity)


# ===============================================
# Extra Analyses
# ===============================================

# # start end price analysis
# start_weighted_price = datac.loc[returns.index].head(1).dot(max_sharpe_weights).squeeze()
# final_weighted_price = datac.loc[returns.index].tail(1).dot(max_sharpe_weights).squeeze()
# (final_weighted_price - start_weighted_price) / start_weighted_price

# # find optimal allocation
# analysis_weights = dict(zip(TICKERS, np.round(max_sharpe_weights, 4)))
# analysis_weights_df = (
#     pd.DataFrame().from_dict(analysis_weights, orient="index").rename(columns={0: "Analysis"})
# )
# prefixed_weights_df = (
#     pd.DataFrame().from_dict(PORTFOLIO, orient="index").rename(columns={0: "Prefixed"})
# )
# prefixed_weights_df.join(analysis_weights_df)
