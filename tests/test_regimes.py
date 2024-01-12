import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from quanttak.regimes import HMMRegime

# Get Data
dt = yf.Ticker("PG")
data = dt.history(period="max")
df = data.loc["2022":"2024", "Open":"Volume"]
df.index = pd.DatetimeIndex(pd.to_datetime(df.index).date)

# Define Constants
TEST_DATA_SIZE = 22
RANDOM_STATE = 1990
ITERATIONS = 1000
TIME_PERIOD = 15
N_STATES = 3


@pytest.fixture
def hmmr():
    return HMMRegime(
        data=df,
        test_data_size=TEST_DATA_SIZE,
        n_states=N_STATES,
        time_period=TIME_PERIOD,
        use_indicators_features=True,
        use_vix_features=True,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
    )


def test_hmm_init(hmmr):
    assert isinstance(hmmr, HMMRegime)


@pytest.fixture
def fitted_hmmr(hmmr):
    hmmr.fit()
    return hmmr


def test_hmm_train(hmmr):
    hmmr.fit()
    # Tests - Train
    assert isinstance(hmmr.predict_train(), pd.DataFrame)
    assert isinstance(hmmr.fnldata_train, pd.DataFrame)
    assert isinstance(hmmr.get_stationary_distribution(), np.ndarray)


def test_hmm_test(fitted_hmmr):
    # Tests - Test
    assert isinstance(fitted_hmmr.predict(), pd.DataFrame)
    assert isinstance(fitted_hmmr.fnldata_test, pd.DataFrame)
