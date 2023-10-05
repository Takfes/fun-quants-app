import os

import finnhub
import pandas as pd
from dotenv import load_dotenv

# https://github.com/Finnhub-Stock-API/finnhub-python

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Setup client
fh = finnhub.Client(api_key=FINNHUB_API_KEY)

# Select tickers
TICKERS = ["UFPT", "CAAP", "SPLP", "MPC"]

# ===============================================
# API functionalities
# ===============================================

# fh.company_eps_estimates("AAPL")
# fh.press_releases("AAPL", _from="2023-01-01", to="2023-08-01")
# fh.news_sentiment("AAPL")

# ===============================================
# Recommendation trends
# ===============================================

recommendations = []

fh.recommendation_trends("CAAP")

for t in TICKERS:
    recommendations.extend(fh.recommendation_trends(t))

dfrec = pd.DataFrame(recommendations)[
    ["symbol", "period", "strongSell", "sell", "hold", "buy", "strongBuy"]
]

dfrec["Total"] = dfrec[["strongSell", "sell", "hold", "buy", "strongBuy"]].sum(axis=1)

# calculate the percentage of each recommendation
dfrec["strongSell"] = dfrec["strongSell"] / dfrec["Total"]
dfrec["sell"] = dfrec["sell"] / dfrec["Total"]
dfrec["hold"] = dfrec["hold"] / dfrec["Total"]
dfrec["buy"] = dfrec["buy"] / dfrec["Total"]
dfrec["strongBuy"] = dfrec["strongBuy"] / dfrec["Total"]

# ===============================================
# Surprises
# ===============================================

fh.company_earnings("CAAP")

surprises = []

for t in TICKERS:
    surprises.extend(fh.company_earnings(t, limit=5))

dfsur = pd.DataFrame(surprises)[
    ["symbol", "period", "actual", "estimate", "surprise", "surprisePercent"]
]
