import pandas as pd
import yfinance as yf

from quanttak.classes.regimes import HMMRegime

# Get Data
dt = yf.Ticker("PG")
data = dt.history(period="max")
df = data.loc["2022":"2024", "Open":"Volume"]
df.index = pd.DatetimeIndex(pd.to_datetime(df.index).date)
df.shape

# Define Constants
TEST_DATA_SIZE = 22
RANDOM_STATE = 1990
ITERATIONS = 1000
TIME_PERIOD = 15
N_STATES = 3

hmmr = HMMRegime(
    data=df,
    test_data_size=TEST_DATA_SIZE,
    n_states=N_STATES,  # ! if change affect regime_names and regime_colors = {"Sideways": "orange", "Bearish": "red", "Bullish": "green"}
    time_period=TIME_PERIOD,
    use_indicators_features=True,
    use_vix_features=True,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
)

hmmr.fit()

hmmr.predict_train()
hmmr.plot_market_regimes_scatter(mode="train")
# hmmr.plot_market_regimes_span(mode="train")
# hmmr.fnldata_train.tail()
print(hmmr.get_stationary_distribution())

hmmr.predict()
hmmr.plot_market_regimes_scatter(mode="test")
# hmmr.plot_market_regimes_span(mode="test")
hmmr.fnldata_test.tail(TEST_DATA_SIZE)["RegimeLabels"].value_counts()
