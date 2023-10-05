import os
import time

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.myio import configure_logging, get_tickers, push_data_to_ohlc

# define CHUNKSIZE
CHUNKSIZE = 100

# define yf download period
START = "2017-01-01"

# Load environment variables from .env file
load_dotenv()
LOG_FILE = os.getenv("LOG_FILE_OHLC")

# configure logging
logger = configure_logging(
    console_output=True, log_to_file=True, log_file_path=LOG_FILE
)

# create timetag with YYYY-MM-DD format
timetag = pd.Timestamp.now().strftime("%Y-%m-%d")

# get tickers
tickersdf = get_tickers()
print(f'Total number of tickers: {len(tickersdf["ticker"].unique())}')

# get unique providers
# provider_list = tickersdf["provider"].unique().tolist()
provider_list = ["ATHEX", "LSE", "NASDAQ", "NYSE"]  # 'BSE', 'SSE', 'TSE'

# ===============================================
# download data historical data
# ===============================================
start = time.perf_counter()

for provider in provider_list:
    tickers = tickersdf.loc[tickersdf["provider"] == provider, "ticker"].tolist()
    # split the tickers into chunks of CHUNKSIZE
    chunks = [tickers[x : x + CHUNKSIZE] for x in range(0, len(tickers), CHUNKSIZE)]
    print(f"\nprovider: {provider} - tickers: {len(tickers)} - chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"provider: {provider} - chunk: {i}/{len(chunks)}")
        try:
            clean_chunk = [x for x in chunk if str(x) != "nan"]
            data_chunk_ = yf.download(clean_chunk, start=START, threads=True)
            # turn data to long format
            data_chunk = data_chunk_.stack().reset_index()
            # rename columns
            data_chunk.columns = [
                "Date",
                "Ticker",
                "AdjClose",
                "Close",
                "High",
                "Low",
                "Open",
                "Volume",
            ]
            # add timetag column
            data_chunk["Timetag"] = timetag
            # add provider column
            data_chunk["Exchange"] = provider
            # push data to ohlc
            push_data_to_ohlc(data_chunk)
        except Exception as e:
            logger.error(f"Error pushing data {provider} to ohlc: {e}")
            continue
    print("")
    print("*🦆*" * 25)
    print("")

end = time.perf_counter()
print(f"Time elapsed: {end - start:.2f} seconds")
