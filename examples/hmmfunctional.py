import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib
import yfinance as yf
from hmmlearn import hmm
from matplotlib.patches import Patch

# Define Constants
RANDOM_STATE = 1990
ITERATIONS = 1000
TIME_PERIOD = 15
N_STATES = 3


# Define Functions
def get_vix_data(df):
    vt = yf.Ticker("^VIX")
    vdata = vt.history(period="max")
    vdata.index = pd.DatetimeIndex(pd.to_datetime(vdata.index).date)
    vf = vdata.loc[df.index]
    return vf


def prep_hmm_data(
    data,
    time_period=15,
    use_indicators_features=True,
    use_vix_features=True,
    plot=False,
):
    from sklearn.preprocessing import StandardScaler

    data = df.copy()
    # Calculate Data Features
    data["Returns"] = data["Close"].pct_change()
    data["ReturnsPeriod"] = data["Close"].pct_change(time_period * 2)
    data["Volatility"] = data["Returns"].rolling(window=time_period).std()
    if use_indicators_features:
        data["RSI"] = talib.RSI(data["Close"], timeperiod=time_period)
        data["ADX"] = talib.ADX(
            data["High"], data["Low"], data["Close"], timeperiod=time_period
        )
    data.dropna(inplace=True)
    if use_vix_features:
        # Calculate VIX Features
        vix = get_vix_data(df)
        vix_columns = vix.columns.tolist()
        vix["Returns_VIX"] = vix["Close"].pct_change()
        vix["ReturnsPeriod_VIX"] = vix["Close"].pct_change(time_period * 2)
        vix["Volatility_VIX"] = vix["Returns_VIX"].rolling(window=time_period).std()
        vix["MA_VIX"] = vix["Close"].rolling(window=time_period).mean()
        vix_features = [x for x in vix.columns if x not in vix_columns]
        vix.dropna(inplace=True)
        # Combine Data
        combo = data.join(vix[vix_features], how="left", rsuffix="_VIX")
        combo_vix_nulls = combo.filter(regex="_VIX$", axis=1).isnull().sum().sum()
        if combo_vix_nulls != 0:
            warnings.warn(
                "VIX Data and Stock Data are not aligned. Continuing without VIX features"
            )
            combo = data.copy()
    else:
        combo = data.copy()
    # Keep Relevant Columns
    subdata = combo.drop(["Open", "High", "Low", "Volume"], axis=1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(subdata)
    scaled_df = pd.DataFrame(scaled_data, index=subdata.index, columns=subdata.columns)
    if plot:
        # Plot Features
        scaled_df.plot()
        plt.show()
        # Create Correlation Heatmap
        corr = scaled_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr, cmap="RdYlGn", center=0, annot=True, fmt=".1f", annot_kws={"size": 9}
        )
        plt.title("Features Correlation Heatmap")
        plt.show()
    return scaled_df.drop("Close", axis=1)


def get_regime_tags(df):
    data = df.copy()
    data["Returns"] = data["Close"].pct_change(15)
    groupobs = data.groupby("MarketRegime")["Returns"].median()
    groupobsdict = groupobs.to_dict()
    max_value = max(groupobsdict.values())
    min_value = min(groupobsdict.values())
    groupobstags = {}
    for key, value in groupobsdict.items():
        if value == max_value:
            groupobstags[key] = "Bullish"
        elif value == min_value:
            groupobstags[key] = "Bearish"
        else:
            groupobstags[key] = "Sideways"
    return groupobstags


def plot_market_regimes_span(df):
    # Ensure the index is a DatetimeIndex for proper plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        print("The DataFrame index should be a DatetimeIndex.")
        return
    # Define colors for different regimes
    regime_colors = {"Sideways": "orange", "Bearish": "red", "Bullish": "green"}
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(15, 8))
    # Plot the close values
    ax.plot(df.index, df["Close"], label="Close Price", color="black", lw=2)
    # Plot background color based on the market regime
    start_idx = df.index[0]
    current_regime = df["RegimeLabels"][0]
    for i, (idx, row) in enumerate(df.iterrows()):
        if row["MarketRegime"] != current_regime or i == len(df) - 1:
            ax.axvspan(start_idx, idx, color=regime_colors[current_regime], alpha=0.3)
            start_idx = idx
            current_regime = row["RegimeLabels"]
    # Customize the plot
    ax.set_title("Market Regimes with Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    legend_patches = [
        Patch(color=color, label=regime) for regime, color in regime_colors.items()
    ]
    plt.legend(handles=legend_patches)
    plt.show()


def plot_market_regimes_scatter(df):
    # Ensure the index is a DatetimeIndex for proper plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        print("The DataFrame index should be a DatetimeIndex.")
        return
    # Define colors for different regimes
    regime_colors = {"Sideways": "orange", "Bearish": "red", "Bullish": "green"}
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(15, 8))
    # Plot each point with color based on the market regime
    for regime, color in regime_colors.items():
        regime_data = df[df["RegimeLabels"] == regime]
        ax.scatter(
            regime_data.index,
            regime_data["Close"],
            color=color,
            label=f"Regime {regime}",
            alpha=0.6,
        )
    # Customize the plot
    ax.set_title("Market Regimes with Close Price (Scatter)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    # Create legend
    plt.legend()
    # Show the plot
    plt.show()


# Get Data
# dt = yf.Ticker("^GSPC")
dt = yf.Ticker("AAPL")
data = dt.history(period="max")
df = data.loc["2010":"2023", "Open":"Volume"]
df.index = pd.DatetimeIndex(pd.to_datetime(df.index).date)
df.shape

# Prep Data
observations = prep_hmm_data(
    df,
    time_period=5,
    use_indicators_features=True,
    use_vix_features=False,
    plot=False,
)
observations.shape

# Initialize Gaussian HMM
model = hmm.GaussianHMM(
    n_components=N_STATES,
    covariance_type="diag",
    n_iter=ITERATIONS,
    random_state=RANDOM_STATE,
)
# Fit the HMM to the observations
model.fit(observations.values)
# Predict the hidden states of the market regimes
hidden_states = model.predict(observations.values)

# Embed hidden states in the data
pltdata = df.loc[observations.index].copy()
pltdata.shape

pltdata["MarketRegime"] = hidden_states
regime_tags = get_regime_tags(pltdata)
pltdata["RegimeLabels"] = pltdata["MarketRegime"].map(regime_tags)

# Plot Market Regimes
plot_market_regimes_scatter(pltdata)
plot_market_regimes_span(pltdata)

# Get the stationary distribution of the hidden states
hidden_states_stats = pd.DataFrame(
    {
        "states": pd.Series(hidden_states).value_counts().sort_index().index,
        "counts": pd.Series(hidden_states).value_counts().sort_index().values,
        "stationary_distribution": model.get_stationary_distribution(),
    }
)
hidden_states_stats["labels"] = hidden_states_stats["states"].map(regime_tags)
hidden_states_stats
