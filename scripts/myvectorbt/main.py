import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

# SL in vectorbt
# https://stackoverflow.com/questions/76328503/how-to-set-a-stoploss-in-vectorbt-based-on-the-number-of-ticks-or-price-per-cont

"""
# ==============================================================
# Download Data
# ==============================================================
"""

master = pd.read_pickle("data/eth_1h_ohlcv.pkl")
df = master.loc["2020-01-01":, :].query('symbol == "ETH/USDT"').copy()
prices = df["close"]

"""
# ==============================================================
# Define Strategies
# ==============================================================
"""


def macd_cross_signal(df):
    dfc = df.copy()
    dfc["macd"], dfc["macdsignal"], dfc["macdhist"] = talib.MACD(
        df.close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    dfc["macd_cross_macdsignal"] = (dfc["macd"] > dfc["macdsignal"]).astype(int)
    dfc["macd_cross_zero"] = (dfc["macd"] > 0).astype(int)
    dfc["macd_cross_signal"] = (
        (dfc["macd_cross_macdsignal"] == 1)
        & (dfc["macd_cross_macdsignal"].rolling(window=12, closed="left").sum() == 0)
        & (dfc["macd_cross_zero"] == 1)
        & (dfc["macd_cross_zero"].rolling(window=3, closed="left").sum() == 0)
    ).astype(bool)
    return dfc["macd_cross_signal"]


def macd_divergence_signal(df):
    # https://medium.com/coinmonks/exit-strategies-for-trading-positions-920f3b95f606
    # https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd#:~:text=one%20strong%20trend.-,Divergences,MACD%20shows%20less%20downside%20momentum.
    dfc = df.copy()
    dfc["macd"], dfc["macdsignal"], dfc["macdhist"] = talib.MACD(
        dfc["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    dfc["close_lower_low"] = (
        dfc["close"] < dfc["close"].rolling(12, closed="left").min()
    ).astype(int)
    dfc["macd_higher_low"] = (
        dfc["macd"] > dfc["macd"].rolling(12, closed="left").max()
    ).astype(int)
    dfc["macd_divergence_signal"] = np.where(
        (dfc["close_lower_low"] == 1) & (dfc["macd_higher_low"] == 1), True, False
    )
    return dfc["macd_divergence_signal"]


"""
# ==============================================================
# Backtest Strategies
# ==============================================================
"""

INITIAL_CASH = 1_000

entries = macd_cross_signal(df)
# entries = macd_divergence_signal(df)
exits = entries.shift(24 * 14).fillna(False)
sum(entries), sum(exits)

# Create a portfolio based on the signals
portfolio = vbt.Portfolio.from_signals(
    prices,
    entries,
    exits,
    init_cash=INITIAL_CASH,
    fees=0.001,  # Set commission (e.g., 0.1% per trade)
    size=INITIAL_CASH / 50,  # Set trade size (e.g., 100 shares per trade)
    freq="1h",  # Set the frequency of the data to hourly
)

"""
# ==============================================================
# Analyze Results
# ==============================================================
"""

# Print performance statistics
pstats = portfolio.stats()
print(pstats)

# Asset flow
portfolio.asset_flow().plot()

# Plot the portfolio value
portfolio.plot().show()

# Print the orders
portfolio.orders.records_readable

# Plot the positions
fig = prices.vbt.plot(trace_kwargs=dict(name="Close"))
portfolio.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)


portfolio.plot(subplots=["cash", "assets", "value"])  # .show_svg()
