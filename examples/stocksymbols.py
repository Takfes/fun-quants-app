import os

from dotenv import load_dotenv
from stocksymbol import StockSymbol

# load api key from .env file
load_dotenv()

api_key = os.getenv("API_KEY_STOCK_SYMBOL")
ss = StockSymbol(api_key)

# get symbol list based on market
symbol_list_market = ss.get_symbol_list(market="GR")  # "us" or "america" will also work
len(symbol_list_market)
sorted(
    list(
        filter(
            None, [(x.get("symbol"), x.get("shortName")) for x in symbol_list_market]
        )
    )
)

# get symbol list based on index
symbol_list_index = ss.get_symbol_list(index="SPX")

# show a list of available market
market_list = ss.market_list

len(market_list)
[x.get("market") for x in market_list]
[x for x in market_list if x.get("market") == "greece"]

# show a list of available index
index_list = ss.index_list

len(index_list)
sorted([x.get("market") for x in index_list])
[x for x in index_list if x.get("market") == "Europe"]
