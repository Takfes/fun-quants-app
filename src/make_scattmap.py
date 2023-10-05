import pprint

import pandas as pd
from matplotlib import pyplot as plt

from src.conf import RISK_FREE_RATE
from src.func import annual_risk_return, find_below_threshold_missingness
from src.myio import query_duckdb

foo = pd.read_pickle("data/metadata.pkl")

# NMS = Nasdaq Global Select Market, large-cap companies
# NGM = Nasdaq Global Market, mid-cap companies
# NCM = Nasdaq Capital Market, early-stage companies
# NYQ = New York Stock Exchange, large-cap companies

boo = foo.loc[
    (foo.exchange.isin(["NCM", "NGM", "NMS", "NYQ"]))
    & (foo.quoteType == "EQUITY")
    & (foo.recommendationKey.isin(["buy", "strong_buy"]))
    & (foo.pegRatio <= 1)
    & (foo.currentRatio >= 1)
    # & (foo.priceToSalesTrailing12Months <= 2)
].copy()

START_DATE = "2021-01-01"
END_DATE = "2023-07-31"

scatdata_query = """
    SELECT TICKER,"DATE",AdjClose
    FROM main.ohlc
    WHERE 1=1
    AND "DATE" >= '{start_date}'
    AND "DATE" <  '{end_date}'
"""

formatted_query = scatdata_query.format(start_date=START_DATE, end_date=END_DATE)
pprint.pprint(formatted_query)

# Get data from database
dbdata = query_duckdb(formatted_query)

# filter tickers by foo
boodata = dbdata.loc[dbdata["Ticker"].isin(boo.ticker), :]

# pivot data to have dates as index and tickers as columns
data = boodata.pivot(index="Date", columns="Ticker", values="AdjClose")

# Track symbols with acceptable missingness
non_missing = find_below_threshold_missingness(data)
# Keep only symbols with acceptable missingness
datac = data[non_missing].copy()
# find returns and drop nas
returns = datac.pct_change().dropna()

rrd = annual_risk_return(returns, risk_free_rate=RISK_FREE_RATE)
rrd.sort_values(by="Sharpe", ascending=False, inplace=True)

rrd_small = rrd.query("Risk < 0.4 & Returns > 0.1 & Returns < 1.8")
# rrd_small.head(10)

scattdata = rrd_small.copy()

# scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(scattdata["Risk"], scattdata["Returns"], alpha=0.5)
# add annotations one by one with a loop
for i in range(len(scattdata)):
    ax.text(
        scattdata.Risk.iloc[i],
        scattdata.Returns.iloc[i],
        scattdata.index[i],
        fontsize=16,
    )
ax.set_xlabel("Risk")
ax.set_ylabel("Returns")
ax.set_title("Risk-Return Scatter Plot")
plt.show()
