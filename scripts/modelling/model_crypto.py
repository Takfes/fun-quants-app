import pandas as pd
import ppscore as pps
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

pd.options.mode.chained_assignment = None  # Disable the SettingWithCopyWarning

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from quanttak.features import FeatureEngineer
from quanttak.targets import make_target_rtns, make_target_tpsl


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


"""
# ==============================================================
# Get Crypto OHLC Data
# ==============================================================
"""

SINCE = "2022-01-01 00:00:00"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
FILEPATH = f"data/crypto_data_{SYMBOL.lower().replace('/','')}_{TIMEFRAME.lower()}_{SINCE.replace(' ','_').replace('-','').replace(':','')}.pkl"

# rawdata = fetch_ohlcv_data(symbol=SYMBOL, timeframe=TIMEFRAME, since=SINCE)
# rawdata.to_pickle(FILEPATH)

rawdata = pd.read_pickle(FILEPATH)

# Get features
fdata = FeatureEngineer(
    rawdata,
    colname_open="open",
    colname_high="high",
    colname_low="low",
    colname_close="close",
    colname_volume="volume",
).generate_all_indicators()

"""
# ==============================================================
# Make Target Column
# ==============================================================
"""

target_rtns = make_target_rtns(rawdata.close, binary=True)

target_tpsl_out = make_target_tpsl(rawdata.close)
rawdata.close.to_clipboard()
target_tpsl_out.to_clipboard()
target_tpsl = target_tpsl_out["hit_binary"]

data = fdata.join(target_rtns, how="inner").join(target_tpsl, how="inner")


target_columns = ["returns", "hitbinary"]
TARGET_VAR_NAME = "returns"  # hitbinary
data["target"] = data[TARGET_VAR_NAME].astype("category")
data.drop(target_columns, axis=1, inplace=True)

data.target.value_counts(normalize=True)

"""
# ==============================================================
# Manual Feature Selection
# ==============================================================
"""

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

"""
# ==============================================================
# Model Training
# ==============================================================
"""

# X = data[featselect]
X = data.drop("target", axis=1)
y = data["target"]

# Define the number of splits for time series cross-validation
n_splits = 10

# Perform time series split
tscv = TimeSeriesSplit(n_splits=n_splits, gap=24 * 5)

"""
# ==============================================================
# GridSearchCV to determine pipeline structure
# ==============================================================
"""

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro"),
    "f1": make_scorer(f1_score, average="macro"),
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
}

# Define the pipeline
pipeline = Pipeline(
    [
        ("interactions", PolynomialFeatures(interaction_only=True)),
        ("scaling", StandardScaler()),
        ("feature_selection", None),
        ("pca", PCA()),
        ("classification", LogisticRegression(max_iter=1000)),
    ]
)

param_grid = [
    {
        "interactions": [PolynomialFeatures(interaction_only=True), None],
        "feature_selection": [
            SelectFromModel(LGBMClassifier(verbose=-100)),
            None,
        ],
        "pca": [PCA(n_components=20), None],
        "classification": [
            LogisticRegression(max_iter=1000),
            LGBMClassifier(verbose=-100),
        ],
    }
]

# Create a GridSearchCV object
pipeline_clone = clone(pipeline)

grid_search = GridSearchCV(
    estimator=pipeline_clone,
    param_grid=param_grid,
    cv=tscv,
    scoring="roc_auc",
    verbose=-100,
)

grid_search = RandomizedSearchCV(
    estimator=pipeline_clone,
    param_distributions=param_grid,
    n_iter=20,
    cv=tscv,
    scoring="roc_auc",
    verbose=-100,
)


# Fit the GridSearchCV object to the data
grid_search.fit(X, y)
print(grid_search.best_params_)
pd.DataFrame(grid_search.cv_results_).to_clipboard()

"""
# ==============================================================
# Manual Training Loop
# ==============================================================
"""

# Define the pipeline
pipeline = Pipeline(
    [
        ("scaling", StandardScaler()),
        # ("feature_selection", SelectKBest(chi2)),
        ("interactions", PolynomialFeatures(interaction_only=True)),
        ("pca", PCA()),
        ("classification", LogisticRegression(max_iter=1000)),
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

resu = pd.concat(
    [x.rename(columns=lambda x: f"{x}_{i}") for i, x in enumerate(metrics, start=1)],
    axis=1,
)
rsg = pd.concat(metrics)
resu_grouped = rsg.groupby(rsg.index).agg(["min", "mean", "std"]).loc[resu.index]
resu_grouped.loc[:, sorted(resu_grouped.columns, key=lambda x: x[1])]

# TODO
# Introduce a function to adjust train & test sets
# 1. Need to account for look-ahead bias
# That is exclude train observations whose target is based on test info
# 2. Need to establish iid observations
# That is exclude multiple train & test observations which may link back to same hit_date info
# You may determine amount of observation in each set affected from the same hit_date
# Impose a distribution on the observations and sample based on that
