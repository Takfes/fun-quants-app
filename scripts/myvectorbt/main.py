import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

# SL in vectorbt
# https://stackoverflow.com/questions/76328503/how-to-set-a-stoploss-in-vectorbt-based-on-the-number-of-ticks-or-price-per-cont


"""
# ==============================================================
# Define Strategies
# ==============================================================
"""


def resampler(df, freq="1h"):
    dfc = df.copy()
    dfc.index = pd.to_datetime(dfc.index)
    dfc = dfc.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return dfc


def macd_cross_entry_signal(df):
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


def macd_cross_exit_signal(df):
    dfc = df.copy()
    dfc["macd"], dfc["macdsignal"], dfc["macdhist"] = talib.MACD(
        df.close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    dfc["macd_cross_macdsignal"] = (dfc["macd"] < dfc["macdsignal"]).astype(int)
    dfc["macd_cross_zero"] = (dfc["macd"] < 0).astype(int)
    dfc["macd_cross_signal"] = (
        (dfc["macd_cross_macdsignal"] == 1)
        & (dfc["macd_cross_macdsignal"].rolling(window=12, closed="left").sum() == 0)
        & (dfc["macd_cross_zero"] == 1)
        & (dfc["macd_cross_zero"].rolling(window=3, closed="left").sum() == 0)
    ).astype(bool)
    return dfc["macd_cross_signal"]


def sma_cross_exit_signal(df):
    dfc = df.copy()
    dfd = resampler(dfc, freq="1d")
    dfd["sma"] = dfd["close"].rolling(20).mean()
    dfd["exit"] = dfd["close"] > dfd["sma"]
    dfc["exit"] = dfd["exit"].reindex(dfc.index).fillna(False)
    return dfc["exit"]


def macd_dvg_entry_signal(df):
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
# Download Data
# ==============================================================
"""

master = pd.read_pickle("data/eth_1h_ohlcv.pkl").drop(columns=["symbol"])
df = master.loc["2020-01-01":, :].copy()  # .query('symbol == "ETH/USDT"').copy()
prices = df["close"]

"""
# ==============================================================
# Backtest Strategies
# ==============================================================
"""

INITIAL_CASH = 1_000

# Define Entry Signals
entries = macd_cross_entry_signal(df)
# entries = macd_dvg_entry_signal(df)

# Define Exit Signals
# exits = entries.shift(24 * 14).fillna(False)
# exits = sma_cross_exit_signal(df)
exits = macd_cross_exit_signal(df)

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

overall_return = (prices[-1] - prices[0]) / prices[0] * 100
print(f"Overall Return: {overall_return:.5f}%")

# Plot the portfolio value
portfolio.plot()

# Plot the trades
portfolio.trades.plot()

# Print the orders
portfolio.orders.records_readable

"""
# ==============================================================
# More results
# ==============================================================
"""

# Various Plots
[x for x in dir(portfolio) if not x.startswith("_")]

portfolio.plot_drawdowns()
portfolio.plot_underwater()
portfolio.plot(subplots=["cash", "assets", "value"])  # .show_svg()
portfolio.asset_flow().plot()
portfolio.cash_flow().plot()
portfolio.order_records
# portfolio.orders.plot()
# portfolio.plot_position_pnl()
# fig = prices.vbt.plot(trace_kwargs=dict(name="Close"))
# portfolio.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)

# Orders JOIN entry/exit signals
# ees = (
#     pd.DataFrame(
#         {"Entries": entries.values.astype(int), "Exits": exits.values.astype(int)},
#         index=entries.index,
#     )
#     .rename_axis("Timestamp")
#     .reset_index()
# )
# orr = portfolio.orders.records_readable
# orr.merge(ees, on="Timestamp", how="right").set_index("Timestamp").query("Entries==1")
