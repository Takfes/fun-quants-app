import pandas as pd
import ppscore as pps
from sklearn.decomposition import PCA
from tqdm import tqdm

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
from quanttak.targets import lookahead_downsampling, make_target_rtns, make_target_tpsl


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
    train_datadict["From"] = X_train.index.min()
    train_datadict["To"] = X_train.index.max()

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
    test_datadict["From"] = y_train.index.min()
    test_datadict["To"] = y_train.index.max()

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

RANDOM_SEED = 1990
SINCE = "2022-01-01 00:00:00"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
FILEPATH = f"data/crypto_data_{SYMBOL.lower().replace('/','')}_{TIMEFRAME.lower()}_{SINCE.replace(' ','_').replace('-','').replace(':','')}.pkl"

# rawdata = fetch_ohlcv_data(symbol=SYMBOL, timeframe=TIMEFRAME, since=SINCE)

# rawdata.to_pickle(FILEPATH)
rawdata = pd.read_pickle(FILEPATH)
rawfeatures = rawdata.columns.tolist()

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

# Returns Column
target_rtns = make_target_rtns(rawdata.close, binary=True, period=1)

# TPSL Column
target_tpsl_out = make_target_tpsl(rawdata.close, tp=0.03, sl=-0.025)
# rawdata.close.to_clipboard()
# target_tpsl_out.to_clipboard()
target_tpsl = target_tpsl_out["hit_binary"]

# Merge Datasets
data = fdata.join(target_rtns, how="inner").join(target_tpsl, how="inner")

# Choose Taret
target_columns = ["returns", "hit_binary"]
TARGET_VAR_NAME = "hit_binary"  # "returns"
data["target"] = data[TARGET_VAR_NAME].astype("category")
data.drop(target_columns, axis=1, inplace=True)

# Review Target
data.target.value_counts(normalize=True)
# data.to_pickle(FILEPATH.replace(".pkl", "_features.pkl"))


"""
# ==============================================================
# Feature Selection
# ==============================================================
"""

# Correlations
corrma = data.corr()[["target"]].rename(columns={"target": "corr"})
corrma["abscorr"] = corrma["corr"].apply(abs)
corrma = (
    corrma.loc[lambda x: x.index.str.contains(r"[A-Z]")]
    .dropna()
    .sort_values(by=["abscorr"], ascending=False)
    .rename_axis("x")
    .reset_index()
)

# PPS Score
ppsdata_out = pps.predictors(data, "target")
ppsdata = ppsdata_out.loc[
    lambda x: (~x["x"].isin(rawfeatures) & (x["baseline_score"] < x["model_score"]))
]

TOP_X_FEATURES = 30
featselect = sorted(
    list(
        set(
            corrma.head(TOP_X_FEATURES)["x"].tolist()
            + ppsdata.head(TOP_X_FEATURES)["x"].tolist()
        )
    )
)
len(featselect)

"""
# ==============================================================
# Model Training
# ==============================================================
"""

X = data[featselect]
# X = data.drop("target", axis=1)

y = data["target"]

# Define the number of splits for time series cross-validation
n_splits = 5

# Perform time series split
tscv = TimeSeriesSplit(n_splits=n_splits)

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
        ("pca", PCA(n_components=20)),
        # ("feature_selection", SelectFromModel(LogisticRegression())),
        ("classification", LogisticRegression(max_iter=1000)),
    ]
)

XX = pipeline.fit_transform(X)

X.shape
XX.shape
pipeline.named_steps["interactions"].n_output_features_
pipeline.named_steps["pca"].explained_variance_ratio_.sum()

param_grid = [
    {
        "interactions": [PolynomialFeatures(interaction_only=True), None],
        # "feature_selection": [
        #     SelectFromModel(LGBMClassifier()),
        #     None,
        # ],
        "pca": [PCA(n_components=20), None],
        "classification": [
            LogisticRegression(max_iter=1000),
            # LGBMClassifier(),
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
    verbose=0,
)

grid_search = RandomizedSearchCV(
    estimator=pipeline_clone,
    param_distributions=param_grid,
    n_iter=20,
    cv=tscv,
    scoring="roc_auc",
    verbose=0,
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
        ("interactions", PolynomialFeatures(interaction_only=True)),
        ("scaling", StandardScaler()),
        ("pca", PCA(n_components=20)),
        # ("feature_selection", SelectFromModel(LogisticRegression())),
        # ("classification", LogisticRegression(max_iter=1000)),
        ("classification", LGBMClassifier(verbosity=0)),
    ]
)

# Iterate over the splits
metrics = []
for train_index, test_index in tqdm(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train_ld, y_train_ld = lookahead_downsampling(
        x_set=X_train, y_set=y_train, steer=target_tpsl_out, fraction_to_sample=0.65
    )
    pipeline_clone = clone(pipeline)
    model = pipeline_clone.fit(X_train_ld, y_train_ld)
    metrics.append(
        evaluate_model(model, X_train_ld, y_train_ld, X_test, y_test)
    )  # ! X_train_ld & y_train_ld or X_train & y_train

# Results splits
results_by_split = pd.concat(
    [x.rename(columns=lambda x: f"{x}_{i}") for i, x in enumerate(metrics, start=1)],
    axis=1,
).T

# Aggregated results
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC"]
rsg = pd.concat(metrics).rename_axis("metrics").reset_index()
results_aggregated = (
    rsg.loc[lambda x: x["metrics"].isin(metrics)]
    .groupby("metrics")
    .agg(["min", "mean", "std"])
)
results_aggregated = results_aggregated.loc[
    metrics, sorted(results_aggregated.columns, key=lambda x: x[1])
]
results_aggregated
