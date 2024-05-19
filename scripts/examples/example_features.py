import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import talib
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split

"""
# ==============================================================
# Download Data
# ==============================================================
"""

timeframe = "1h"
symbolsinput = [
    # "ADA",
    "ETH",
    # "BNB",
    # "BTC",
    # "VET",
    # "SOL",
    # "XRP",
    # "LINK",
    # "AVAX",
    # "MATIC",
    # "DOT",
    # "TON",
    # "DOGE",
    # "UNI",
    # "APT",
    # "SHIB",
]
symbols = sorted([f"{x}/USDT" for x in symbolsinput])

# Download Data
data = []
# for symbol in tqdm(symbols):
#     print(f"Downloading {symbol}")
#     temp = fetch_ohlcv_data(symbol, timeframe, since="2020-01-01 00:00:00")
#     temp.insert(0, "symbol", symbol)
#     data.append(temp)

master = pd.concat(data)
master.to_pickle("data/eth_1h_ohlcv.pkl")
master.to_csv("data/eth_1h_ohlcv.csv")

master = pd.read_pickle("data/eth_1h_ohlcv.pkl")

"""
# ==============================================================
# Process Data
# ==============================================================
"""

df = master.loc["2023-01-01":, :].query('symbol == "ETH/USDT"').copy()
df.shape

# df["simple_returns"] = df["close"].pct_change()
# df["log_returns"] = np.log(df.close / df.close.shift(1))

# # Assess Returns over a period of one, two, three etc days
# lookahead_periods = [24, 48, 72, 96, 120, 144, 168]  # , 192, 216, 240]
# for lp in lookahead_periods:
#     df[f"forward_returns_{lp}"] = (df["close"].shift(-lp) - df["close"]) / df["close"]

# # Descriptives
# (df.filter(like="forward_returns").dropna().apply(abs).describe().T)

# # Select "target column"
# target_returns = "forward_returns_168"
# return_columns = df.filter(like="forward_returns").columns.tolist()
# return_columns_drop = [x for x in return_columns if x != target_returns]
# df = df.drop(return_columns_drop, axis=1).rename(columns={target_returns: "target"})
# df.dropna(inplace=True)


"""
# ==============================================================
# Generate Signals and Feature
# ==============================================================
"""


def macd_cross_signal(df):
    dfc = df.copy()
    dfc["macd"], dfc["macdsignal"], dfc["macdhist"] = talib.MACD(
        df.close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    dfc["macd_cross_macdsignal"] = (dfc["macd"] > dfc["macdsignal"]).astype(int)
    dfc["macd_cross_zero"] = (dfc["macd"] > 0).astype(int)
    dfc["macd_cross_signal"] = (
        (dfc["macd_cross_macdsignal"] == 1)
        & (dfc["macd_cross_macdsignal"].rolling(window=12, closed="left").sum() == 0)
        & (dfc["macd_cross_zero"] == 1)
        & (dfc["macd_cross_zero"].rolling(window=3, closed="left").sum() == 0)
    ).astype(int)
    return dfc["macd_cross_signal"]


df["macd_cross_signal"] = macd_cross_signal(df)
df["macd_cross_signal"].value_counts()


def macd_divergence_signal(df):
    # https://medium.com/coinmonks/exit-strategies-for-trading-positions-920f3b95f606
    # https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd#:~:text=one%20strong%20trend.-,Divergences,MACD%20shows%20less%20downside%20momentum.
    dfc = df.copy()
    dfc["macd"], dfc["macdsignal"], dfc["macdhist"] = talib.MACD(
        dfc["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    dfc["close_lower_low"] = (
        dfc["close"] < dfc["close"].rolling(12, closed="left").min()
    ).astype(int)
    dfc["macd_higher_low"] = (
        dfc["macd"] > dfc["macd"].rolling(12, closed="left").max()
    ).astype(int)
    dfc["macd_divergence_signal"] = np.where(
        (dfc["close_lower_low"] == 1) & (dfc["macd_higher_low"] == 1), 1, 0
    )
    return dfc["macd_divergence_signal"]


df["macd_divergence_signal"] = macd_divergence_signal(df)
df["macd_divergence_signal"].value_counts()


def locate_local_maxima(df, sensitivity=10):
    dfc = df.copy()
    dfc["local_maxima"] = (
        (
            dfc["close"][::-1]
            > dfc["close"][::-1].rolling(sensitivity, closed="left").max()
        )[::-1]
        & (dfc["close"] > dfc["close"].rolling(sensitivity, closed="left").max())
    ).astype(int)
    return dfc["local_maxima"]


df["local_maxima"] = locate_local_maxima(df, sensitivity=24 * 14)
df["local_maxima"].value_counts()


"""
# ==============================================================
# Plotting
# ==============================================================
"""

plt.figure(figsize=(12, 8))
plt.plot(df.index, df["close"], label="close")

buy_signals = df[df["macd_cross_signal"] == 1]
plt.scatter(
    buy_signals.index,
    buy_signals["close"] - 150,
    label="Buy Signal",
    marker="^",
    color="green",
    s=250,
)

buy_signals2 = df[df["macd_divergence_signal"] == 1]
plt.scatter(
    buy_signals2.index,
    buy_signals2["close"] - 150,
    label="Buy Signal",
    marker="^",
    color="blue",
    s=250,
)

target_signals = df[df["local_maxima"] == 1]

plt.scatter(
    target_signals.index,
    target_signals["close"] + 150,
    label="Buy Signal",
    marker="v",
    color="red",
    s=250,
)

plt.legend(loc="upper left")
plt.title("Signal Strategy")
plt.show()


"""
# ==============================================================
# Build ML Model
# ==============================================================
"""

# # Add All features from the ta library
# ta.add_all_ta_features(
#     df, open="open", high="high", low="low", close="close", volume="volume"
# )

# Add Pattern Recognition Features
# pattern_features = talib.get_function_groups()["Pattern Recognition"]
# for feature in pattern_features:
#     df[feature.lower()] = getattr(talib, feature)(
#         df["open"], df["high"], df["low"], df["close"]
#     )

pattern_features = talib.get_function_groups()["Pattern Recognition"]
new_cols = {
    feature.lower(): getattr(talib, feature)(
        df["open"], df["high"], df["low"], df["close"]
    )
    for feature in pattern_features
}
df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

# Remove NAs
# Capture columns with larger missingness than the target column
features_drop = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .loc[lambda x: x > x.target]
    .index.tolist()
)
len(features_drop)
df = df.drop(features_drop, axis=1).dropna()
df.shape

# Sample Dataframe for training
SAMPLE_SIZE = 1000
top = (
    df["target"]
    .dropna()
    .sort_values(ascending=False)
    .loc[lambda x: x >= 0.05]
    .sample(SAMPLE_SIZE)
)

mid = (
    df["target"]
    .dropna()
    .sort_values(ascending=False)
    .loc[lambda x: (x < 0.05) & (x > 0.0)]
    .sample(SAMPLE_SIZE)
)

bot = (
    df["target"]
    .dropna()
    .sort_values(ascending=False)
    .loc[lambda x: x < 0.0]
    .sample(SAMPLE_SIZE)
)

[x.shape[0] for x in [top, mid, bot]]
sampled_index = [
    item for sublist in [x.index for x in [top, mid, bot]] for item in sublist
]
len(sampled_index)

# Create Dataframe for training
target_value = top.min()
df = df.loc[sampled_index, :].copy()
df.shape

df["target_binary"] = (df["target"] >= target_value).astype(int)
target_columns = [x for x in df.columns.tolist() if "target" in x]
non_target_columns = [x for x in df.columns.tolist() if x not in target_columns]
df = df[non_target_columns + target_columns].copy()
df["target_binary"].value_counts()

# Split the data into training and testing sets
X = df.drop(["symbol", "target_binary", "target"], axis=1)
# X = df[
#     [
#         "volume_adi",
#         "volume_obv",
#         "volume_nvi",
#         "volume_vpt",
#         "trend_visual_ichimoku_b",
#         "volume_cmf",
#         "volatility_atr",
#         "trend_adx",
#         "momentum_pvo_signal",
#     ]
# ]
y = df["target_binary"]

# Train Model


def evaluate_classifier(clf, X_test, y_test):
    try:
        # Predict the labels for the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = clf.score(X_test, y_test)
        print("Accuracy:", accuracy)

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

        # Calculate and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Calculate and plot the ROC curve
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (AUC = {:.2f})".format(auc))
        plt.show()
    except Exception as e:
        print(e)


# Random Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Create the LightGBM classifier
clf = lgb.LGBMClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
evaluate_classifier(clf, X_test, y_test)
print("Accuracy:", accuracy)


# Get feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": clf.feature_importances_}
)

# Sort feature importance in descending order
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
# feature_importance.to_clipboard()


# Perform time series split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create the LightGBM classifier
    clf = lgb.LGBMClassifier(verbosity=0)

    # Train the classifier
    clf.fit(X_train, y_train)

    print(y_train.value_counts(normalize=True))
    # Evaluate the classifier
    evaluate_classifier(clf, X_test, y_test)
    # accuracy = clf.score(X_test, y_test)
    # print("Accuracy:", accuracy)
