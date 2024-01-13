import pandas as pd
import ppscore as pps
import yfinance as yf

pd.options.mode.chained_assignment = None  # Disable the SettingWithCopyWarning

from sklearn.decomposition import PCA
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from quanttak.features import FeatureEngineer
from quanttak.targets import make_target_rtns, make_target_tpsl

START = "2015-01-01"
END = "2024-01-02"
SYMBOL = "AAPL"

# Get OHLC Data - Single symbol
yraw = yf.download(SYMBOL, start=START, end=END)
ydata = yraw.rename(columns=lambda x: x.lower().replace(" ", ""))

# Get features
fdata = FeatureEngineer(
    ydata,
    colname_open="open",
    colname_high="high",
    colname_low="low",
    colname_close="close",
    colname_volume="volume",
).generate_all_indicators()

# Make Target Columns
target_pctc = make_target_rtns(ydata.adjclose, binary=True)
target_tpsl = make_target_tpsl(ydata.adjclose)["hitbinary"]
data = fdata.join(target_pctc, how="inner").join(target_tpsl, how="inner")

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
featselect = list(
    set(
        corrma.head(TOP_X_FEATURES)["x"].tolist()
        + ppsdat.head(TOP_X_FEATURES)["x"].tolist()
    )
)

# Machine Learning
X = data[featselect]
y = data[TARGET_VAR_NAME]

# Import necessary library
# Define the number of splits for time series cross-validation
n_splits = 5

# Perform time series split
tscv = TimeSeriesSplit(n_splits=n_splits)

# Iterate over the splits
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Import necessary libraries
# Define the pipeline
pipeline = Pipeline(
    [
        ("scaling", StandardScaler()),  # Feature scaling
        ("pca", PCA(n_components=10)),  # PCA for dimensionality reduction
        (
            "interactions",
            PolynomialFeatures(interaction_only=True),
        ),  # Feature interactions
        ("classification", LogisticRegression(max_iter=1000)),  # Classification model
    ]
)

model = pipeline.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate classification measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:")
    print("TN:", tn, "FP:", fp)
    print("FN:", fn, "TP:", tp)


evaluate_model(model, X_test, y_test)
