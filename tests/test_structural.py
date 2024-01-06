import pandas as pd
import pytest

from quanttak.classes.structural import FMRegressor

ASSETS = ["PG"]
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

fm = FMRegressor(ASSETS, START_DATE, END_DATE)
fm.fit()
pd.DataFrame(fm.results)


@pytest.fixture
def factor_model():
    ASSETS = ["PG"]
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"
    return FMRegressor(ASSETS, START_DATE, END_DATE)


def test_get_asset_data(factor_model):
    assert isinstance(factor_model.get_asset_data(), pd.DataFrame)


def test_get_factor_data(factor_model):
    assert isinstance(factor_model.get_factor_data(), pd.DataFrame)


def test_process_data(factor_model):
    temp = factor_model.process_data()
    assert isinstance(temp, pd.DataFrame)
    for col in ["Mkt_RF", "SMB", "HML", "MOM", "RMW", "CMA", "Excess_Returns"]:
        assert col in temp.columns


def test_fit(factor_model):
    factor_model.fit()
    assert isinstance(factor_model.results, dict)
