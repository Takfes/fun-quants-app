import pandas as pd
import ppscore as pps

pd.options.mode.chained_assignment = None  # Disable the SettingWithCopyWarning

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quanttak.data import fetch_ohlcv_data
from quanttak.features import FeatureEngineer
from quanttak.targets import make_target_rtns, make_target_tpsl

SINCE = "2023-01-01 00:00:00"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"


def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_datadict = {}
    test_datadict = {}

    # Make predictions on train set
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate classification measures for train set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(
        y_train, y_train_pred
    ).ravel()

    # Calculate classification measures for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Store results for train set
    train_datadict["Accuracy"] = train_accuracy
    train_datadict["Precision"] = train_precision
    train_datadict["Recall"] = train_recall
    train_datadict["F1"] = train_f1
    train_datadict["ROC"] = train_roc_auc
    train_datadict["TP"] = train_tp
    train_datadict["TN"] = train_tn
    train_datadict["FP"] = train_fp
    train_datadict["FN"] = train_fn
    train_datadict["Size"] = train_tp + train_tn + train_fp + train_fn

    # Store results for test set
    test_datadict["Accuracy"] = test_accuracy
    test_datadict["Precision"] = test_precision
    test_datadict["Recall"] = test_recall
    test_datadict["F1"] = test_f1
    test_datadict["ROC"] = test_roc_auc
    test_datadict["TP"] = test_tp
    test_datadict["TN"] = test_tn
    test_datadict["FP"] = test_fp
    test_datadict["FN"] = test_fn
    test_datadict["Size"] = test_tp + test_tn + test_fp + test_fn

    return (
        pd.concat(
            [
                pd.DataFrame().from_dict(train_datadict, orient="index"),
                pd.DataFrame().from_dict(test_datadict, orient="index"),
            ],
            ignore_index=True,
            axis=1,
        ).rename(columns={0: "Train", 1: "Test"})
        # .assign(Delta=lambda x: (x["Test"] - x["Train"]) / x["Train"])
    )


# Get Crypto OHLC Data
rawdata = fetch_ohlcv_data(symbol=SYMBOL, timeframe=TIMEFRAME, since=SINCE)

# Get features
fdata = FeatureEngineer(
    rawdata,
    colname_open="open",
    colname_high="high",
    colname_low="low",
    colname_close="close",
    colname_volume="volume",
).generate_all_indicators()

# Make Target Columns
target_tpsl = make_target_tpsl(rawdata.close)["hitbinary"]
target_rtns = make_target_rtns(rawdata.close, binary=True)
data = fdata.join(target_rtns, how="inner")

# Feature Selection
TARGET_VAR_NAME = "returns"

# Correlations
corrma = data.corr()[[TARGET_VAR_NAME]].rename(columns={TARGET_VAR_NAME: "corr"})
corrma["abscorr"] = corrma["corr"].apply(abs)
corrma = (
    corrma.loc[lambda x: x.index.str.contains(r"[A-Z]")]
    .dropna()
    .sort_values(by=["abscorr"], ascending=False)
    .rename_axis("x")
    .reset_index()
)

# PPS Score
data[f"{TARGET_VAR_NAME}_category"] = data[TARGET_VAR_NAME].astype("category")
ppsdat = pps.predictors(data, f"{TARGET_VAR_NAME}_category")
ppsdat = ppsdat.loc[lambda x: x["x"].str.contains(r"[A-Z]")]

TOP_X_FEATURES = 15
featselect = sorted(
    list(
        set(
            corrma.head(TOP_X_FEATURES)["x"].tolist()
            + ppsdat.head(TOP_X_FEATURES)["x"].tolist()
        )
    )
)

# Machine Learning
X = data[featselect]
y = data[TARGET_VAR_NAME]

# Import necessary library
# Define the number of splits for time series cross-validation
n_splits = 10

# Perform time series split
tscv = TimeSeriesSplit(n_splits=n_splits)

# Define the pipeline
pipeline = Pipeline(
    [
        ("scaling", StandardScaler()),  # Feature scaling
        # ("pca", PCA(n_components=5)),  # PCA for dimensionality reduction
        # (
        #     "interactions",
        #     PolynomialFeatures(interaction_only=True),
        # ),  # Feature interactions
        (
            "classification",
            LogisticRegression(max_iter=1000),
        ),  # Classification model
    ]
)

# Iterate over the splits
metrics = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipeline_clone = clone(pipeline)
    model = pipeline_clone.fit(X_train, y_train)
    metrics.append(evaluate_model(model, X_train, y_train, X_test, y_test))

resu = pd.concat(metrics, axis=0)
resu_grouped = resu.groupby(resu.index).mean()
