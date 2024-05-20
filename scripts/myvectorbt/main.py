import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

"""
# ==============================================================
# References
# ==============================================================
"""

# vectorbt : https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_signals
# sizing in vectorbt : https://www.marketcalls.in/python/mastering-vectorbt-position-sizing-code-snippets-part-4-python-tutorial.html
# SL in vectorbt : https://stackoverflow.com/questions/76328503/how-to-set-a-stoploss-in-vectorbt-based-on-the-number-of-ticks-or-price-per-cont
# sl_stop & tp_stop in vectorbt : https://quantnomad.com/using-sl-and-pt-in-backtesting-in-python-with-vectrobt/

"""
# ==============================================================
# Define Strategies
# ==============================================================
"""


# def resampler(df, freq="1h"):
#     dfc = df.copy()
#     dfc.index = pd.to_datetime(dfc.index)
#     dfc = dfc.resample(freq).agg(
#         {
#             "open": "first",
#             "high": "max",
#             "low": "min",
#             "close": "last",
#             "volume": "sum",
#         }
#     )
#     return dfc


# def sma_cross_exit_signal(df):
#     dfc = df.copy()
#     dfd = resampler(dfc, freq="1d")
#     dfd["sma"] = dfd["close"].rolling(20).mean()
#     dfd["sma_cross_exit_signal"] = dfd["close"] > dfd["sma"]
#     dfc["sma_cross_exit_signal"] = (
#         dfd["sma_cross_exit_signal"].reindex(dfc.index).fillna(False)
#     )
#     return dfc["sma_cross_exit_signal"]


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
df = master.loc["2021-06-01":, :].copy()  # .query('symbol == "ETH/USDT"').copy()
prices = df["close"]

"""
# ==============================================================
# Setting Up Backtesting
# ==============================================================
"""

INITIAL_CASH = 1_000
SIZE = 0.10
# SIZE_TYPE = "value"  # fixed cash value
# SIZE_TYPE = "amount"  # fixed number of units of the instrument
SIZE_TYPE = "percent"  # percentage of the current portfolio value
CLOSE_LAST_POSITION = True

# Define Entry Signals
entries = macd_cross_entry_signal(df)
# entries = macd_dvg_entry_signal(df)
# entries = pd.concat([macd_cross_entry_signal(df), macd_dvg_entry_signal(df)], axis=1)
# entries.sum()

# Define Exit Signals
# exits = entries.shift(24 * 14).fillna(False)
# exits = sma_cross_exit_signal(df)
exits = macd_cross_exit_signal(df)
# exits = pd.concat([macd_cross_exit_signal(df), sma_cross_exit_signal(df)], axis=1)
# exits.sum()

# Force to close the last position
exits[-1] = True if CLOSE_LAST_POSITION else exits[-1]
sum(entries), sum(exits)

# Create a portfolio based on the signals
portfolio = vbt.Portfolio.from_signals(
    prices,
    entries,
    exits,
    init_cash=INITIAL_CASH,
    fees=0.001,
    size=SIZE,
    size_type=SIZE_TYPE,
    # sl_stop=0.02,
    # tp_stop=0.45,
    freq="1h",
)

# Print performance statistics
pstats = portfolio.stats()
print(pstats)

"""
# ==============================================================
# Analyze Results
# ==============================================================
"""

# Plot the portfolio value
portfolio.plot(subplots=["trades", "trade_pnl", "cash", "cum_returns", "drawdowns"])

# Print the orders
portfolio.orders.records_readable

# Print the trades
portfolio.trades.records_readable

"""
# ==============================================================
# More results
# ==============================================================
"""

# Various Plots
[x for x in dir(portfolio) if not x.startswith("_")]

# Calculations
# overall_return = (prices[-1] - prices[0]) / prices[0] * 100
# print(f"Overall Return: {overall_return:.5f}%")

trr = portfolio.trades.records_readable
trr.query("PnL > 0")["Return"].describe()
trr.query("PnL < 0")["Return"].describe()

cashflow = (portfolio.cash_flow().cumsum() + INITIAL_CASH).rename("Cash").reset_index()
trademore = trr.merge(cashflow, left_on="Exit Timestamp", right_on="timestamp").drop(
    columns="timestamp"
)

portfolio.cash_flow().loc[lambda x: x != 0]
portfolio.cash_flow().loc[lambda x: x != 0].cumsum()

# More Plots
portfolio.plot_cash()
portfolio.trades.plot()
# (portfolio.cash_flow().cumsum() + INITIAL_CASH).plot()
portfolio.plot_drawdowns()
portfolio.plot_underwater()
portfolio.plot(
    subplots=["cash", "assets", "value", "cum_returns", "trades"]
)  # .show_svg()
portfolio.asset_flow().plot()
portfolio.cash_flow().plot()
portfolio.order_records
# portfolio.orders.plot()
# portfolio.plot_position_pnl()
# fig = prices.vbt.plot(trace_kwargs=dict(name="Close"))
# portfolio.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)

# Orders JOIN entry/exit signals
ees = (
    pd.DataFrame(
        {"Entries": entries.values.astype(int), "Exits": exits.values.astype(int)},
        index=entries.index,
    )
    .rename_axis("Timestamp")
    .reset_index()
)
# orr = portfolio.orders.records_readable
# orr.merge(ees, on="Timestamp", how="right").set_index("Timestamp").query("Entries==1")

# Check valid entry/exit pairs
latest_entry = None
position = False
invalid_entries = []
invalid_exits = []
valid_pairs = []

for idx, row in ees.iterrows():
    if row["Entries"] == 1:
        if position:
            invalid_entries.append(row["Timestamp"])
        else:
            latest_entry = row["Timestamp"]
            valid_pairs.append([latest_entry, None])
            position = True
    if row["Exits"] == 1:
        if position:
            valid_pairs[-1][1] = row["Timestamp"]
            position = False
            latest_entry = None
        else:
            invalid_exits.append(row["Timestamp"])

len(valid_pairs), len(invalid_entries), len(invalid_exits)
assert len(valid_pairs) == pstats["Total Trades"]
assert (
    len([x for x in valid_pairs if x[1] is not None]) == pstats["Total Closed Trades"]
)

pstats
portfolio.orders.records_readable
valid_pairs

"""
# ==============================================================
# Test Multiple Strategies
# ==============================================================
"""
# Test multiple strategies
# windows = [(10, 50), (20, 100), (50, 200)]

# entries = pd.DataFrame(
#     {
#         f"{s1}_{s2}": vbt.MA.run(prices, window=s1).ma_crossed_above(
#             vbt.MA.run(prices, window=s2).ma
#         )
#         for s1, s2 in windows
#     }
# )
# exits = pd.DataFrame(
#     {
#         f"{s1}_{s2}": vbt.MA.run(prices, window=s1).ma_crossed_below(
#             vbt.MA.run(prices, window=s2).ma
#         )
#         for s1, s2 in windows
#     }
# )

# # Ensure the signals are aligned with the price data
# entries = entries.reindex(prices.index, fill_value=False)
# exits = exits.reindex(prices.index, fill_value=False)
