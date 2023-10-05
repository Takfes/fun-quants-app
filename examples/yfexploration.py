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

df_a.xs("AAPL", level=1, axis=1).columns

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
ticker_symbol = "AAPL"  # "AAPL"

# Create the Ticker object
stock = yf.Ticker(ticker_symbol)

# Display the Ticker object
[x for x in dir(stock) if not x.startswith("_")]

# get historical market data
hd = stock.history(period="max", actions=True)

# misc
stock.financials
stock.dividends
stock.dividends.tail(1).squeeze() / stock.get_fast_info()["lastPrice"]

stock.get_info()
stock.get_earnings_forecast()
stock.get_analyst_price_target()
stock.analyst_price_target()
stock.get_recommendations()
stock.recommendations()
stock.get_history_metadata()
stock.get_isin()
stock.get_news()


stock.get_info()["address1"]
stock.get_info()["city"]
stock.get_info()["state"]
stock.get_info()["zip"]
stock.get_info()["country"]
stock.get_info()["phone"]
stock.get_info()["website"]
stock.get_info()["industry"]
stock.get_info()["industryDisp"]
stock.get_info()["sector"]
stock.get_info()["sectorDisp"]
stock.get_info()["longBusinessSummary"]
stock.get_info()["fullTimeEmployees"]
stock.get_info()["companyOfficers"]
stock.get_info()["auditRisk"]
stock.get_info()["boardRisk"]
stock.get_info()["compensationRisk"]
stock.get_info()["shareHolderRightsRisk"]
stock.get_info()["overallRisk"]
stock.get_info()["governanceEpochDate"]
stock.get_info()["compensationAsOfEpochDate"]
stock.get_info()["maxAge"]
stock.get_info()["priceHint"]
stock.get_info()["previousClose"]
stock.get_info()["open"]
stock.get_info()["dayLow"]
stock.get_info()["dayHigh"]
stock.get_info()["regularMarketPreviousClose"]
stock.get_info()["regularMarketOpen"]
stock.get_info()["regularMarketDayLow"]
stock.get_info()["regularMarketDayHigh"]
stock.get_info()["dividendRate"]
stock.get_info()["dividendYield"]
stock.get_info()["exDividendDate"]
stock.get_info()["payoutRatio"]
stock.get_info()["fiveYearAvgDividendYield"]
stock.get_info()["beta"]
stock.get_info()["trailingPE"]
stock.get_info()["forwardPE"]
stock.get_info()["volume"]
stock.get_info()["regularMarketVolume"]
stock.get_info()["averageVolume"]
stock.get_info()["averageVolume10days"]
stock.get_info()["averageDailyVolume10Day"]
stock.get_info()["bid"]
stock.get_info()["ask"]
stock.get_info()["bidSize"]
stock.get_info()["askSize"]
stock.get_info()["marketCap"]
stock.get_info()["fiftyTwoWeekLow"]
stock.get_info()["fiftyTwoWeekHigh"]
stock.get_info()["priceToSalesTrailing12Months"]
stock.get_info()["fiftyDayAverage"]
stock.get_info()["twoHundredDayAverage"]
stock.get_info()["trailingAnnualDividendRate"]
stock.get_info()["trailingAnnualDividendYield"]
stock.get_info()["currency"]
stock.get_info()["enterpriseValue"]
stock.get_info()["profitMargins"]
stock.get_info()["floatShares"]
stock.get_info()["sharesOutstanding"]
stock.get_info()["sharesShort"]
stock.get_info()["sharesShortPriorMonth"]
stock.get_info()["sharesShortPreviousMonthDate"]
stock.get_info()["dateShortInterest"]
stock.get_info()["sharesPercentSharesOut"]
stock.get_info()["heldPercentInsiders"]
stock.get_info()["heldPercentInstitutions"]
stock.get_info()["shortRatio"]
stock.get_info()["shortPercentOfFloat"]
stock.get_info()["impliedSharesOutstanding"]
stock.get_info()["bookValue"]
stock.get_info()["priceToBook"]
stock.get_info()["lastFiscalYearEnd"]
stock.get_info()["nextFiscalYearEnd"]
stock.get_info()["mostRecentQuarter"]
stock.get_info()["earningsQuarterlyGrowth"]
stock.get_info()["netIncomeToCommon"]
stock.get_info()["trailingEps"]
stock.get_info()["forwardEps"]
stock.get_info()["pegRatio"]
stock.get_info()["lastSplitFactor"]
stock.get_info()["lastSplitDate"]
stock.get_info()["enterpriseToRevenue"]
stock.get_info()["enterpriseToEbitda"]
stock.get_info()["52WeekChange"]
stock.get_info()["SandP52WeekChange"]
stock.get_info()["lastDividendValue"]
stock.get_info()["lastDividendDate"]
stock.get_info()["exchange"]
stock.get_info()["quoteType"]
stock.get_info()["symbol"]
stock.get_info()["underlyingSymbol"]
stock.get_info()["shortName"]
stock.get_info()["longName"]
stock.get_info()["firstTradeDateEpochUtc"]
stock.get_info()["timeZoneFullName"]
stock.get_info()["timeZoneShortName"]
stock.get_info()["uuid"]
stock.get_info()["messageBoardId"]
stock.get_info()["gmtOffSetMilliseconds"]
stock.get_info()["currentPrice"]
stock.get_info()["targetHighPrice"]
stock.get_info()["targetLowPrice"]
stock.get_info()["targetMeanPrice"]
stock.get_info()["targetMedianPrice"]
stock.get_info()["recommendationMean"]
stock.get_info()["recommendationKey"]
stock.get_info()["numberOfAnalystOpinions"]
stock.get_info()["totalCash"]
stock.get_info()["totalCashPerShare"]
stock.get_info()["ebitda"]
stock.get_info()["totalDebt"]
stock.get_info()["quickRatio"]
stock.get_info()["currentRatio"]
stock.get_info()["totalRevenue"]
stock.get_info()["debtToEquity"]
stock.get_info()["revenuePerShare"]
stock.get_info()["returnOnAssets"]
stock.get_info()["returnOnEquity"]
stock.get_info()["grossProfits"]
stock.get_info()["freeCashflow"]
stock.get_info()["operatingCashflow"]
stock.get_info()["earningsGrowth"]
stock.get_info()["revenueGrowth"]
stock.get_info()["grossMargins"]
stock.get_info()["ebitdaMargins"]
stock.get_info()["operatingMargins"]
stock.get_info()["financialCurrency"]
stock.get_info()["trailingPegRatio"]
