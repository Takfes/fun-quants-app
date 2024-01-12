import datetime
import logging
from typing import List

import pandas as pd
from rich.logging import RichHandler


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


def convert_timestamp(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1000)


def convert_datetime_to_ms(datetime_str: str) -> int:
    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    timestamp = int(datetime_obj.timestamp() * 1000)
    return timestamp


def convert_date(x):
    return datetime.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")


def find_below_threshold_missingness(
    data: pd.DataFrame, threshold: float = 0.0
) -> List:
    return (
        (data.isnull().sum() / data.shape[0])
        .loc[lambda x: x <= threshold]
        .index.tolist()
    )
