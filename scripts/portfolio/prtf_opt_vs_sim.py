import time

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from src.conf import RISK_FREE_RATE
from src.func import (
    annual_risk_return,
    find_below_threshold_missingness,
    plot_portfolios,
    rebalance_weights,
)
from src.myio import histdata_query, query_duckdb

sns.set_style("darkgrid")

# ===============================================
# Get data
# ===============================================

RANDOM_SEED = 1990
PULL_DATA_FROM_DATABASE = True
SIMULATION_REDUCED_UNIVERSE = False
SIMULATION_NUMBER_OF_PORTFOLIOS = 50_000
START_DATE = "2021-01-01"
END_DATE = "2023-07-31"
THRESHOLD = 0.05

# tickersdf = get_tickers()
# TICKERS = tickersdf[tickersdf.provider == "ATHEX"].ticker.tolist()
TICKERS = ["ELF", "CAAP", "CER.L", "MCK", "IPEL.L", "ARLP"]  # "XOM", "NVDA",
# TICKERS = [
#     "CVX",
#     "RS",
#     "CW",
#     "MET",
#     "FSK",
#     "TAC",
#     "ADI",
#     "PPC",
#     "UFPI",
#     "COP",
#     "MPC",
#     "CMC",
#     "HRB",
# ]

print(f"{20*'='} PULL DATA {20*'='}")
if PULL_DATA_FROM_DATABASE:
    # Query Data from Database
    formatted_query = histdata_query.format(
        tickers=tuple(TICKERS), start_date=START_DATE, end_date=END_DATE
    )
    # pprint.pprint(formatted_query)
    # Get data from database
    dbdata = query_duckdb(formatted_query)
    # pivot data to have dates as index and tickers as columns
    data = dbdata.pivot(index="Date", columns="Ticker", values="AdjClose")
else:
    # Download data
    rdata = yf.download(TICKERS, start=START_DATE, end=END_DATE, actions=True)
    # Keep only the Adj Close column
    data = rdata["Adj Close"].copy()

# dbdata.sum() == data.sum()
# dbdata.equals(data)

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
# plot_assets_density(returns)

# # create correlation heatmap
# corr = returns.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(
#     corr,
#     mask=mask,
#     cmap=cmap,
#     vmax=1,
#     vmin=-1,
#     center=0,
#     square=True,
#     linewidths=0.5,
#     cbar_kws={"shrink": 0.5},
# )

# TODO INCORPORATE DIVIDENDS AND FOUNDAMENTALS FILTERS
# TODO CLEAN OUTLIER RETURNS

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
print(f"> Number of days in data: {datac.shape[0]}")
print(f"> Number of days with returns: {returns.shape[0]}")
print(
    f"> Number of days dropped due to missingness {datac.shape[0] - returns.shape[0]}"
)

# ===============================================
# Optimization
# ===============================================
print(f"\n{20*'='} OPTIMIZATION {20*'='}")

# Set number of assets
NOA = returns.shape[1]
returns_to_use_optimization = returns.copy()

# Calculate expected returns and the covariance matrix
expected_returns = returns_to_use_optimization.mean()
covariance_matrix = returns_to_use_optimization.cov()

# Initialize weights
weights = np.array([1.0 / NOA] * NOA)  # Starting with equal weights

# Constraints
constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1,
}  # Weights sum to 1

# Boundaries for weights (can be adjusted depending on requirements)
bounds = tuple((0, 0.75) for _ in range(NOA))


# Objective Function (We want to maximize Sharpe Ratio, so we minimize the negative Sharpe Ratio)
def objective(weights):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return -(
        portfolio_return / portfolio_stddev
    )  # Negative Sharpe Ratio (assuming risk-free rate is 0 for simplicity)


# Solve the optimization problem
solution = minimize(
    objective, weights, method="SLSQP", bounds=bounds, constraints=constraints
)

# Get the optimal asset weights
optimal_weights_sol = solution.x
optimal_weights = rebalance_weights(optimal_weights_sol)

# Calculate Risk and Return for the optimized portfolio
opt_port_ret = returns_to_use_optimization.dot(optimal_weights)
opt_port_summary = annual_risk_return(
    opt_port_ret.to_frame(), risk_free_rate=RISK_FREE_RATE
)

# find total returns
opt_weighted_returns = opt_port_ret
opt_simple_returns = opt_weighted_returns.sum()
opt_compounded_returns = (opt_weighted_returns + 1).prod() - 1

assert len(returns_to_use_optimization.columns) == optimal_weights.shape[0]
opt_weight_allocation = dict(
    zip(returns_to_use_optimization.columns, np.round(optimal_weights, 4))
)

opt_weight_allocation_nz = {k: v for k, v in opt_weight_allocation.items() if v != 0}
opt_weight_allocation_nz_sorted = dict(
    sorted(opt_weight_allocation_nz.items(), key=lambda item: item[1], reverse=True)
)


# print(f"Simple Returns: {opt_simple_returns:.2%}")
# print(f"Compounded Returns: {opt_compounded_returns:.2%}")
# print(f"Sharpe Ratio Portfolio: {opt_port_summary['Sharpe'].squeeze():.2%}")
# TODO WHY RETURNS DO NOT MATCH?
print(f"Portfolio Summary (Annualized Metrics):\n{opt_port_summary.T}")
print(f"\nWeight Allocation:\n{opt_weight_allocation_nz_sorted}")

# ===============================================
# Simulation
# ===============================================
print(f"\n{20*'='} SIMULATION {20*'='}")

# Set number of portfolios
NOP = SIMULATION_NUMBER_OF_PORTFOLIOS

# Set number of assets
if SIMULATION_REDUCED_UNIVERSE:
    print(
        f"Reducing SIMULATION universe to {len(opt_weight_allocation_nz_sorted)} assets"
    )
    reduced_tickers = list(opt_weight_allocation_nz_sorted.keys())
    NOA = len(reduced_tickers)
    returns_to_use_simulation = returns[reduced_tickers].copy()
else:
    NOA = returns.shape[1]
    returns_to_use_simulation = returns.copy()

# Set weights
np.random.seed(RANDOM_SEED)
matrix = np.random.random(NOP * NOA).reshape(NOP, NOA)
weights = matrix / matrix.sum(axis=1).reshape(-1, 1)
all(weights.sum(axis=1, keepdims=True)) == 1

# Calculate Risk and Return for initial assets
assets_summary = annual_risk_return(
    returns_to_use_simulation, risk_free_rate=RISK_FREE_RATE
)

# Generate random portfolios
sim_port_ret = returns_to_use_simulation.dot(weights.T)

# Calculate Risk and Return for the random portfolios
start = time.perf_counter()
portfolios_summary = annual_risk_return(sim_port_ret, risk_free_rate=RISK_FREE_RATE)
end = time.perf_counter()
print(f"Time taken: {end - start:.2f} seconds")

# find max sharpe ratio
max_sharpe_idx = portfolios_summary.Sharpe.idxmax()
max_sharpe_weights = weights[int(max_sharpe_idx), :]
max_sharpe = portfolios_summary.loc[max_sharpe_idx]

# find total returns
sim_weighted_returns = returns_to_use_simulation.dot(max_sharpe_weights)
sim_simple_returns = sim_weighted_returns.sum()
sim_compounded_returns = (sim_weighted_returns + 1).prod() - 1

assert len(returns_to_use_simulation.columns) == max_sharpe_weights.shape[0]
sim_weight_allocation = dict(
    zip(returns_to_use_simulation, np.round(max_sharpe_weights, 4))
)

# print(f"Simple Returns: {sim_simple_returns:.2%}")
# print(f"Compounded Returns: {sim_compounded_returns:.2%}")
# print(f"Sharpe Ratio Portfolio: {max_sharpe['Sharpe']:.2%}")
# TODO WHY RETURNS DO NOT MATCH?
print(f"Portfolio Summary (Annualized Metrics):\n{max_sharpe.to_frame()}")
print(f"\nWeight Allocation:\n{sim_weight_allocation}")

# ===============================================
# Comparison of Optimization vs Simulation
# ===============================================
print(f"\n{20*'='} VISUALIZATION {20*'='}")
# Compare optimization vs simulation
wopt = pd.DataFrame().from_dict(
    opt_weight_allocation_nz, orient="index", columns=["Optimization Weights"]
)
wsim = pd.DataFrame().from_dict(
    sim_weight_allocation, orient="index", columns=["Simulation Weights"]
)
wdf = wopt.join(wsim, how="outer").sort_values(
    by="Optimization Weights", ascending=False
)

# Plot Portfolios
plot_portfolios(
    assets_data=assets_summary,
    portfolio_data=portfolios_summary,
    optimization_data=opt_port_summary,
    annotate=True,
)
