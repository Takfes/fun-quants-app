import datetime
import os

import ccxt
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from stocksymbol import StockSymbol


def fetch_ohlcv_data(symbol: str, timeframe: str, since: str = "2023-01-01 00:00:00"):
    datetime_obj = datetime.datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
    since_ms = int(datetime_obj.timestamp() * 1000)
    exchange = ccxt.binance()

    all_ohlcv_data = []
    while True:
        ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, since_ms)
        if len(ohlcv_data) == 0:
            break

        all_ohlcv_data.extend(ohlcv_data)
        last_fetched_timestamp = ohlcv_data[-1][0]
        since_ms = last_fetched_timestamp + (1 * 60000)  # Increment by 1 minute in ms

    df = pd.DataFrame(
        all_ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def get_fundamentals_yf(symbol: str):
    metadata = [
        "shortName",  # A shorter or abbreviated name for the company.
        "longName",  # The full name of the company.
        "exchange",  # The stock exchange where the stock is traded.
        "quoteType",  # Indicates the type of financial instrument, e.g., stock, option, mutual fund.
        "sectorDisp",
        # -----------------------------------------------
        "marketCap",  # The total market value of a company's outstanding shares of stock. It's calculated by multiplying the stock's price by the total number of outstanding shares.
        "currentPrice",  # The current price of the stock.
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        # -----------------------------------------------
        "52WeekChange",  # The percentage change in a stock's price over the past 52 weeks.
        "beta",  # A measure of a stock's volatility in relation to the overall market. A beta greater than 1 indicates higher volatility, while less than 1 indicates lower volatility.
        # -----------------------------------------------
        "trailingEps",  # The sum of a company's earnings per share for the trailing 12 months.
        "trailingPE",  # Price-to-Earnings ratio based on the past 12 months of earnings.
        # -----------------------------------------------
        "bookValue",
        "priceToBook",  # The ratio of a company's stock price to its book value per share.
        # -----------------------------------------------
        "earningsQuarterlyGrowth",
        "earningsGrowth",  # The percentage growth in a company's earnings over a specific period.
        "revenueGrowth",  # The percentage growth in a company's revenue over a specific period.
        # -----------------------------------------------
        "volume",
        "averageDailyVolume10Day",
        # -----------------------------------------------
        "dividendRate",
        "dividendYield",  # The annual dividend payment divided by the stock's current market price. It indicates the income generated from an investment in the stock.
        "recommendationKey",  # The consensus recommendation of analysts for the stock.
    ]
    data = {}
    data["symbol"] = symbol
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
        for metric in metadata:
            data[f"{metric}"] = info.get(metric, None)
    except Exception as e:
        print(f"Error for {symbol=}")
        print(e)
        for metric in metadata:
            data[f"{metric}"] = None
    finally:
        return data


def get_fundamentals_av(symbol):
    load_dotenv()
    api_key = os.getenv("API_KEY_ALPHA_VANTAGE")
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    return response.json()


def get_markets(simplify=True):
    response = requests.get("https://stock-symbol.herokuapp.com/api/market/all")
    json_data = response.json()
    if simplify:
        datasets = []
        for item in json_data["data"]:
            temp = pd.DataFrame(item.get("index"))
            temp["market"] = item.get("market")
            temp["abbreviation"] = item.get("abbreviation")
            datasets.append(temp)
        data = (
            pd.concat(datasets).sort_values(by=["market", "id"]).reset_index(drop=True)
        )
        return data
    else:
        return json_data


def get_symbols(market=None, index=None, symbols_only: str = None, simplify=True):
    load_dotenv()
    api_key = os.getenv("API_KEY_STOCK_SYMBOL")
    ss = StockSymbol(api_key)
    results = ss.get_symbol_list(market=market, index=index, symbols_only=symbols_only)
    if simplify:
        return sorted(
            list(
                filter(
                    None,
                    [x.get("symbol") for x in results],
                )
            )
        )
    else:
        return results
