import os

import duckdb
import pandas as pd
from dotenv import load_dotenv


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
