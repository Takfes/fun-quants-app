import warnings
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib
import yfinance as yf
from hmmlearn import hmm
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler


# Define Class
@dataclass
class HMMRegime:
    def __init__(
        self,
        data,
        test_data_size,
        n_states=3,  # ! if change affect regime_names and regime_colors = {"Sideways": "orange", "Bearish": "red", "Bullish": "green"}
        time_period=15,
        use_indicators_features=True,
        use_vix_features=True,
        iterations=1000,
        random_state=1990,
    ):
        self.data = data
        self.test_data_size = test_data_size
        self.n_states = n_states
        self.time_period = time_period
        self.use_indicators_features = use_indicators_features
        self.use_vix_features = use_vix_features
        self.scaler = StandardScaler()
        self.random_state = random_state
        self.iterations = iterations
        self.train = self.data.iloc[: -self.test_data_size]
        self.test = self.data.iloc[-self.test_data_size :]

    def __post_init__(self):
        pass

    def plot_correlation_heatmap(self, df):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            df,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 9},
        )
        plt.title("Features Correlation Heatmap")
        return plt

    def get_vix_data(self, df):
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period="max")
        vix_data.index = pd.DatetimeIndex(pd.to_datetime(vix_data.index).date)
        vix = vix_data.loc[df.index]
        return vix

    def preprocess_data_vix(self, df):
        vix = self.get_vix_data(df)
        vix_columns = vix.columns.tolist()
        vix["Returns_VIX"] = vix["Close"].pct_change()
        vix["ReturnsPeriod_VIX"] = vix["Close"].pct_change(self.time_period * 2)
        vix["Volatility_VIX"] = (
            vix["Returns_VIX"].rolling(window=self.time_period).std()
        )
        vix["MA_VIX"] = vix["Close"].rolling(window=self.time_period).mean()
        self.vix_features = [x for x in vix.columns if x not in vix_columns]
        self.vix = vix.dropna()
        return self.vix

    def preprocess_data_inp(self, df):
        data = df.copy()
        inp_columns = data.columns.tolist()
        data["Returns"] = data["Close"].pct_change()
        data["ReturnsPeriod"] = data["Close"].pct_change(self.time_period * 2)
        data["Volatility"] = data["Returns"].rolling(window=self.time_period).std()
        if self.use_indicators_features:
            data["RSI"] = talib.RSI(data["Close"], timeperiod=self.time_period)
            data["ADX"] = talib.ADX(
                data["High"], data["Low"], data["Close"], timeperiod=self.time_period
            )
        self.inp_features = [x for x in data.columns if x not in inp_columns]
        self.inp = data.dropna()
        return self.inp

    def preprocess_hmm_data(self, df):
        data = self.preprocess_data_inp(df)
        if self.use_vix_features:
            vix = self.preprocess_data_vix(df)
            combo = data.join(vix[self.vix_features], how="left", rsuffix="_VIX")
            combo_vix_nulls = combo.filter(regex="_VIX$", axis=1).isnull().sum().sum()
            if combo_vix_nulls != 0:
                warnings.warn(
                    "Input Data and VIX Data not aligned... Continuing without VIX Data"
                )
                combo = data.copy()
        else:
            combo = data.copy()
        subdata = combo.drop(["Open", "High", "Low", "Volume"], axis=1)
        self.feat_corr_mat = subdata.corr()
        subdata.drop("Close", axis=1, inplace=True)
        return subdata

    def get_regime_tags(self, df):
        data = df.copy()
        data["Returns"] = data["Close"].pct_change(self.time_period)
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

    def get_stationary_distribution(self):
        try:
            return self.model.get_stationary_distribution()
        except Exception as e:
            print(e)

    def fit(self):
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.iterations,
            random_state=self.random_state,
        )
        train_features = self.preprocess_hmm_data(self.train)
        train_processed = self.scaler.fit_transform(train_features)
        self.train_processed = pd.DataFrame(
            train_processed,
            index=train_features.index,
            columns=train_features.columns,
        )
        self.model.fit(self.train_processed.values)

    def predict_train(self):
        self.hidden_states_train = self.model.predict(self.train_processed.values)
        fnldata = self.train.loc[self.train_processed.index].copy()
        fnldata["MarketRegime"] = self.hidden_states_train
        regime_tags = self.get_regime_tags(fnldata)
        fnldata["RegimeLabels"] = fnldata["MarketRegime"].map(regime_tags)
        self.fnldata_train = fnldata
        return self.fnldata_train

    def predict(self):
        data_features = self.preprocess_hmm_data(self.data)
        data_processed = self.scaler.transform(data_features)
        self.data_processed = pd.DataFrame(
            data_processed,
            index=data_features.index,
            columns=data_features.columns,
        )
        self.hidden_states_test = self.model.predict(self.data_processed.values)
        fnldata = self.data.loc[self.data_processed.index].copy()
        fnldata["MarketRegime"] = self.hidden_states_test
        regime_tags = self.get_regime_tags(fnldata)
        fnldata["RegimeLabels"] = fnldata["MarketRegime"].map(regime_tags)
        self.fnldata_test = fnldata
        return self.fnldata_test

    def plot_market_regimes_scatter(self, mode="test"):
        if mode == "test":
            df = self.fnldata_test.copy()
            train_end_date = self.fnldata_train.index[
                -1
            ]  # Get the last date of the training data
        else:
            df = self.fnldata_train.copy()
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
        # Place vertical dotted lines every month
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to monthly
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %Y")
        )  # Set format for the dates
        ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Set minor ticks to monthly
        plt.xticks(rotation=45)
        for month in df["Close"].resample("M").mean().index:  # Loop through each month
            plt.axvline(x=month, color="grey", linestyle="dotted", linewidth=0.5)
        # If mode=="test", place an additional line to signify the last point of the "train" chunk
        if mode == "test":
            plt.axvline(
                x=train_end_date,
                color="black",
                linestyle="--",
                label="End of Train Data",
            )
        # Customize the plot
        ax.set_title(
            f"Market Regimes with Close Price {mode.upper()} | {df.shape[0]} rows"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        # Create legend
        plt.legend()
        # Show the plot
        plt.show()

    def plot_market_regimes_span(self, mode="test"):
        if mode == "test":
            df = self.fnldata_test.copy()
            train_end_date = self.fnldata_train.index[
                -1
            ]  # Get the last date of the training data
        else:
            df = self.fnldata_train.copy()
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
                ax.axvspan(
                    start_idx, idx, color=regime_colors[current_regime], alpha=0.3
                )
                start_idx = idx
                current_regime = row["RegimeLabels"]
        # Place vertical dotted lines every month
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to monthly
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %Y")
        )  # Set format for the dates
        ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Set minor ticks to monthly
        plt.xticks(rotation=45)
        for month in df["Close"].resample("M").mean().index:  # Loop through each month
            plt.axvline(x=month, color="grey", linestyle="dotted", linewidth=0.5)
        # If mode=="test", place an additional line to signify the last point of the "train" chunk
        if mode == "test":
            plt.axvline(
                x=train_end_date,
                color="black",
                linestyle="--",
                label="End of Train Data",
            )
        # Customize the plot
        ax.set_title(
            f"Market Regimes with Close Price {mode.upper()} | {df.shape[0]} rows"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        legend_patches = [
            Patch(color=color, label=regime) for regime, color in regime_colors.items()
        ]
        plt.legend(handles=legend_patches)
        plt.show()
