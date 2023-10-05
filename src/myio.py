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

histdata_query = """
    SELECT TICKER,"DATE",AdjClose
    FROM main.ohlc
    WHERE 1=1
    AND Ticker IN {tickers}
    AND "DATE" >= '{start_date}'
    AND "DATE" <  '{end_date}'
    ORDER BY 1,2
    """


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
    # Execute the query and fetch the result as a DataFrame
    df = conn.execute(sql_query).fetch_df()
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


def fetch_fundamentals(symbol: str):
    from src.conf import metadata

    ticker = yf.Ticker(symbol)
    info = ticker.get_info()
    data = {}
    data[f"ticker"] = symbol
    for metric in metadata:
        data[f"{metric}"] = info.get(metric, None)
    return data
