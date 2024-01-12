import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# pandas options to display more rows and columns
pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)


# ===============================================
# define functions
# ===============================================


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std, (returns - RISK_FREE_RATE) / std


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in np.arange(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return, _ = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix
        )
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


# ===============================================
# download data
# ===============================================

# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
ALLOWED_MISSINGNESS_THRESHOLD = 0.0
MINIMUM_WEIGHT = 0.02
NUMBER_PORTFOLIO_SIMULATIONS = 25000
RISK_FREE_RATE = 0.0178

ticker_major = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "TSLA",
    "FB",
    "BRK.B",
    "JNJ",
    "JPM",
    "PG",
]
ticker_promising = [
    "ZM",
    "PLTR",
    "LMND",
    "NIO",
    "SNOW",
    "CRWD",
    "BYND",
    "DKNG",
    "PLUG",
    "U",
]
ticker_dividend = ["RIO", "MO", "JPM", "MCD", "HD"]

ticker_list = list(
    set(
        ticker_major + ticker_promising + ticker_dividend + ticker_remote_work_portfolio
    )
)

start_date = "2023-01-01"
end_date = "2023-07-31"


# ===============================================
# data preparation
# ===============================================
# download data
data = yf.download(ticker_list, start=start_date, end=end_date)
# keep only the adjusted close
dataclose = data["Adj Close"].copy()
# list to track tickers with less than 20% missingness
ticker_candidates = (
    (dataclose.isnull().sum() / len(dataclose))
    .loc[lambda x: x <= ALLOWED_MISSINGNESS_THRESHOLD]
    .index.tolist()
)
# keep only tickers with less than 20% missingness
datacand = dataclose[ticker_candidates].copy()
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


# Solve the optimization problem
solution = minimize(
    objective, weights, method="SLSQP", bounds=bounds, constraints=constraints
)

# Get the optimal asset weights
optimal_weights = solution.x

# Get allocation for each asset
pre_allocation = {
    k: v for k, v in zip(ticker_candidates, optimal_weights) if v > MINIMUM_WEIGHT
}

# reassigned weights so that add up to 1
valid_weights = np.array(list(pre_allocation.values()))
total_weight_to_reassign = 1 - sum(valid_weights)
weights_complement = (valid_weights * total_weight_to_reassign) / sum(valid_weights)
final_weights = valid_weights + weights_complement
allocation = {k: v for k, v in zip(ticker_candidates, final_weights)}
expected_returns_filtered = {k: expected_returns.get(k) for k in allocation.keys()}
covariance_matrix_filtered = returns[list(allocation.keys())].cov()


results, weights = random_portfolios(
    NUMBER_PORTFOLIO_SIMULATIONS, expected_returns, covariance_matrix, RISK_FREE_RATE
)

max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[0])

results[:, max_sharpe_idx]
results[:, min_vol_idx]

# ===============================================
# all in one the chatgpt way
# ===============================================


def fetch_data(tickers, start_date="2010-01-01", end_date="2020-01-01"):
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data


def portfolio_annual_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_annual_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std


def optimize_portfolio(tickers, risk_free_rate=0.01):
    data = fetch_data(tickers)
    returns = data.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(tickers)

    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initializing weights equally
    initial_weights = [1.0 / num_assets for asset in tickers]

    solution = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return solution.x


# Sample tickers for demonstration
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
tickers = ticker_candidates
optimal_weights = optimize_portfolio(tickers)
