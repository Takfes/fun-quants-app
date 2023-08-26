import datetime
import logging
import os
from typing import List

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from rich.logging import RichHandler
from tqdm import tqdm


def configure_logging(console_output=True, log_to_file=False, log_file_path=None):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create a formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Handlers
    handlers = []
    if console_output:
        console_handler = RichHandler(level=logging.INFO)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    if log_to_file:
        if log_file_path is None:
            raise ValueError("Log file path must be provided if log_to_file is True")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    # Add handlers to the logger
    for handler in handlers:
        logger.addHandler(handler)
    return logger


def query_duckdb(sql_query):
    # Load environment variables from .env file
    load_dotenv()
    # Get database path from .env file
    db_path = os.getenv("DATABASE_NAME")
    # Check if DUCKDB_PATH is set in .env file
    if not db_path:
        raise ValueError("DUCKDB_PATH not set in .env file")
    # Connect to the DuckDB database
    conn = duckdb.connect(db_path)
    # Use pandas to execute the query and fetch the result as a DataFrame
    df = pd.read_sql(sql_query, conn)
    # Close the connection
    conn.close()
    return df


def push_data_to_ohlc(df: pd.DataFrame):
    # Load environment variables from .env file
    load_dotenv()
    # Get values from .env
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    SCHEMA_NAME = os.getenv("SCHEMA_NAME")
    OHLC_TABLE = os.getenv("OHLC_TABLE")
    # Connect to DuckDB
    conn = duckdb.connect(DATABASE_NAME)
    # Insert the DataFrame into the 'ohlc' table
    # Using "INSERT OR IGNORE" to ensure unique constraint is not violated
    conn.register("temp_df", df)
    conn.execute(
        f"""
    INSERT OR IGNORE INTO {SCHEMA_NAME}.{OHLC_TABLE} SELECT * FROM temp_df
    """
    )
    # Close the connection
    conn.close()


def get_tickers(filter: str = None):
    temp = pd.read_excel("data/se_tickers.xlsx")
    return temp
    # if filter is None:
    #     return temp["ticker"].unique().tolist()
    # else:
    #     plist = temp["provider"].unique().tolist()
    #     if filter not in plist:
    #         raise ValueError(f"Filter {filter} not found. Must be on of {plist}")
    #     else:
    #         return temp.loc[temp["provider"] == filter, "ticker"].unique().tolist()


metrics = {
    "trailingPE": "low_is_better",
    "priceToBook": "low_is_better",
    "priceToSalesTrailing12Months": "low_is_better",
    "profitMargins": "high_is_better",
    "returnOnEquity": "high_is_better",
    "returnOnAssets": "high_is_better",
    "currentRatio": "high_is_better",
    "quickRatio": "high_is_better",
    "debtToEquity": "low_is_better",
    "earningsGrowth": "high_is_better",
    "revenueGrowth": "high_is_better",
    "dividendYield": "high_is_better",
    "payoutRatio": "low_is_better",
    "marketCap": "high_is_better",
    "beta": "low_is_better",
    "operatingMargins": "high_is_better",
    "freeCashflow": "high_is_better",
}


def fetch_fundamentals(tickers):
    fundamentals_ = []
    for ticker_symbol in tqdm(tickers):
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.get_info()
        # Extract the relevant metrics
        data = {}
        for metric, preference in metrics.items():
            data[f"{metric}_{preference}"] = info.get(metric, None)
            data["ticker"] = ticker_symbol
        # Append the data to the DataFrame
        fundamentals_.append(data)
        fundamentals_df = pd.DataFrame(fundamentals_).set_index("ticker")
        # Fill missing values with the mean
        fundamentals = fundamentals_df.fillna(fundamentals_df.mean())
        # Rank the metrics
        for column, preference in metrics.items():
            col_name = f"{column}_{preference}"
            if preference == "high_is_better":
                fundamentals[f"{column}_rank"] = fundamentals[col_name].rank(
                    ascending=False
                )
            else:
                fundamentals[f"{column}_rank"] = fundamentals[col_name].rank(
                    ascending=True
                )
        fundamentals["overall_score"] = fundamentals[
            [f"{column}_rank" for column in metrics.keys()]
        ].sum(axis=1)
    return fundamentals.sort_values(by="overall_score")
