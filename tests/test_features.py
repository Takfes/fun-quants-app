import pandas as pd
import pytest
import yfinance

from quanttak.features import FeatureEngineer

SYMBOL = "AAPL"
START = "2023-01-01"
END = "2024-01-02"

data = yfinance.download(SYMBOL, start=START, end=END)


@pytest.fixture
def feature_engineer():
    return FeatureEngineer(data)


def test_generate_all_indicators(feature_engineer):
    assert isinstance(feature_engineer.generate_all_indicators(), pd.DataFrame)
