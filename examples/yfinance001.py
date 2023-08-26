import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from newspaper import Article
from scipy.optimize import minimize

# pandas options to display more rows and columns
pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)

# ===============================================
# batch download data
# ===============================================
ticker_a = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "FB", "BRK.B", "JNJ", "JPM", "PG"]
ticker_b = ["ZM", "PLTR", "LMND", "NIO", "SNOW", "CRWD", "BYND", "DKNG", "PLUG", "U"]
ticker_d = ["RIO", "MO", "JPM", "MCD", "HD"]

# Download data
df_a = yf.download(ticker_a, period="max")

# turn data into long format keeping the ticker symbol as an extra column
df_a_long = (
    df_a.stack(level=1).reset_index(level=1).rename(columns={"level_1": "ticker"})
)

# keep only the adjusted close
df_a_long_adjclose = df_a_long[["ticker", "Adj Close"]]

# ===============================================
# Get Indices Data
# ===============================================

indices = [
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^NDX",
    "^RUT",
    "^VIX",
    "^FTSE",
    "^FTW5000",
    "^N225",
    "^GDAXI",
    "^FCHI",
    "^STOXX50E",
]

xticks = yf.Ticker("^NDX")
# [x for x in dir(xticks) if not x.startswith("_")]
xticks.history_metadata
xticks.get_info()
xticks.history()

# ===============================================
# Get ETF Data
# ===============================================

xticks = yf.Ticker("0JFF.L")
xticks.history_metadata
xticks.get_info()


# ===============================================
# fetch detailed data per ticker
# ===============================================
# Define the ticker symbol
ticker_symbol = "RIO"  # "AAPL"

# Create the Ticker object
stock = yf.Ticker(ticker_symbol)

# Display the Ticker object
[x for x in dir(stock) if not x.startswith("_")]

# get historical market data
hd = stock.history(period="max")

# misc
stock.financials
stock.dividends
stock.dividends.tail(1).squeeze() / stock.get_fast_info()["lastPrice"]

stock.get_fast_info()
stock.get_fast_info()["shares"]
stock.get_fast_info()["lastPrice"]
stock.get_fast_info()["tenDayAverageVolume"]
stock.get_fast_info()["fiftyDayAverage"]
stock.get_fast_info()["yearChange"]

stock.get_info()["dividendRate"]
stock.get_info()["dividendYield"]
stock.get_info()["dividendGrowth"]
stock.get_info()["lastDividendValue"]
stock.get_info()["beta"]
stock.get_info()["volume"]
stock.get_info()["currentPrice"]
stock.get_info()["sharesOutstanding"]
stock.get_info()["marketCap"]
stock.get_info()["industryDisp"]
stock.get_info()["priceToBook"]
stock.get_info()["52WeekChange"]
stock.get_info()["bookValue"]
stock.get_info()["debtToEquity"]
stock.get_info()["ebitda"]
stock.get_info()["ebitdaMargins"]
stock.get_info()["enterpriseToEbitda"]
stock.get_info()["enterpriseToRevenue"]
stock.get_info()["enterpriseValue"]

# get info
info = stock.get_info()
info.keys()
infodict = {
    k: v
    for k, v in info.items()
    if k
    not in [
        "address1",
        "city",
        "state",
        "zip",
        "country",
        "phone",
        "website",
        "longBusinessSummary",
        "companyOfficers",
    ]
}
infodf = pd.DataFrame(infodict, index=[0]).T.sort_index()
market_cap = infodf.loc["marketCap"].squeeze()
outstanding_shares = infodf.loc["sharesOutstanding"].squeeze()
market_cap / outstanding_shares

# get news
news = stock.get_news()
news[0]["title"]
news[0]["link"]
convert_date(news[0]["providerPublishTime"])

article = Article(url=news[0]["link"])
article.download()
article.parse()
article.title
article.text
article.nlp()
article.keywords
article.summary
[x for x in dir(article) if not x.startswith("_")]


# ===============================================
# Create Equal Weighted Portfolio
# ===============================================

# # Get number of assets
# noa = returns.shape[1]

# # Create weights for equal weighted portfolio
# weights = [1 / noa for _ in range(noa)]

# # Calculate portfolio returns
# ewp = returns.copy()
# ewp["EWP"] = returns.dot(weights)
# summary = annual_risk_return(returns)

# # Clean summary
# summaryc = summary.copy()
# # summaryc = summaryc.loc[(summaryc.Returns <= 1) & (summaryc.Returns > 0)].copy()
# # summaryc = summaryc.loc[(summaryc.Returns > 0.15) & (summaryc.Risk < 0.65)].copy()
# # summaryc["Ratio"] = summaryc.Returns / summaryc.Risk
# summaryc.shape

# # summaryc = summaryc.sort_values(by="Ratio", ascending=False).head(30)
# print(f"Number of symbols in summary: {summaryc.shape[0]}")
# plot_summary(summaryc, annotate=True)

# ..........

# NOA = 10
# symbols_to_include_in_portfolio = (
#     summaryc.loc[summaryc.index != "EWP"]
#     .sort_values(by="Ratio", ascending=False)
#     .head(NOA)
#     .index.tolist()
# )
# returns_to_use = returns[symbols_to_include_in_portfolio].copy()

# sns.heatmap(
#     returns_to_use.corr(), cmap="Reds", annot=True, annot_kws={"size": 5}, fmt=".2%", vmax=0.6
# )
