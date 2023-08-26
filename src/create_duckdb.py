import os

import duckdb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get values from .env
DATABASE_NAME = os.getenv("DATABASE_NAME")
SCHEMA_NAME = os.getenv("SCHEMA_NAME")
OHLC_TABLE = os.getenv("OHLC_TABLE")
META_TABLE = os.getenv("META_TABLE")

# Connect to DuckDB
conn = duckdb.connect(DATABASE_NAME)

# Create the 'ohlc' table
print(f'Creating the "{OHLC_TABLE}" table in the "{DATABASE_NAME}" database')

conn.execute(
    f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{OHLC_TABLE} (
    Date DATE,
    Ticker VARCHAR,
    AdjClose FLOAT,
    Close FLOAT,
    High FLOAT,
    Low FLOAT,
    Open FLOAT,
    Volume INTEGER,
    Timetag DATE,
    Exchange VARCHAR,
    UNIQUE (Ticker, Date)
)
"""
)

# # Create the 'meta' table for metadata
# conn.execute(
#     f"""
# CREATE TABLE IF NOT EXISTS {META_TABLE} (
#     Ticker VARCHAR PRIMARY KEY,
#     Metadata VARCHAR
# )
# """
# )

# Close the connection
conn.close()
