import os
import time

import pandas as pd
from dotenv import load_dotenv
from src.myio import configure_logging, fetch_fundamentals, get_tickers
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
LOG_FILE = os.getenv("LOG_FILE_META")

# configure logging
logger = configure_logging(
    console_output=True, log_to_file=True, log_file_path=LOG_FILE
)

# get tickers
tickersdf = get_tickers()
print(f'Total number of tickers: {len(tickersdf["ticker"].unique())}')

# get providers of interest
# provider_list = tickersdf["provider"].unique().tolist()
# provider_list = ["ATHEX", "NASDAQ", "NYSE"]  # "LSE", 'BSE', 'SSE', 'TSE'
provider_list = ["LSE"]

# filter tickers by provider
dataset = tickersdf.loc[
    tickersdf["provider"].isin(provider_list), ["provider", "ticker"]
]

# create a list of tuples with (provider, ticker)
tuples_list = [tuple(x) for x in dataset.to_numpy()]

# ===============================================
# download metadata
# ===============================================

metadata = []
start = time.perf_counter()

for provider, ticker in tqdm(tuples_list):
    try:
        data = fetch_fundamentals(ticker)
        metadata.append(data)
    except Exception as e:
        logger.error(f"Error pushing data {ticker} to meta: {e}")
        continue

end = time.perf_counter()
print(f"Time elapsed: {end - start:.2f} seconds")

metadata = pd.DataFrame(metadata)

metadata.to_pickle("data/metadata-lse.pkl")
metadata.to_excel("data/metadata-lse.xlsx")
