import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from quanttak.data import get_fundamentals_yf, get_symbols

"""
# ==============================================================
# Define Functions
# ==============================================================
"""


def remove_outliers(data, iqr_factor=1.5):
    """
    Removes outliers from a DataFrame based on the IQR method.

    This function calculates the IQR (Interquartile Range) for each numeric column in the DataFrame.
    Any value that is less than Q1 - 1.5 * IQR or greater than Q3 + 1.5 * IQR is considered an outlier.
    All rows containing at least one outlier value are removed from the DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame from which outliers should be removed.

    Returns:
    pandas.DataFrame: A new DataFrame with the same columns as the input, but with outliers removed.
    """
    df = data.copy()
    # Create an empty list to store indices of rows with outliers
    indices_to_remove = []
    # Loop through each column in the DataFrame
    for column in df.select_dtypes(include=["float", "int"]).columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # Calculate the outlier range
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        # Get the indices of outliers and append to the list
        outlier_indices = df[
            (df[column] < lower_bound) | (df[column] > upper_bound)
        ].index
        indices_to_remove.extend(outlier_indices)
    # Remove duplicates from the list
    indices_to_remove = list(set(indices_to_remove))
    # Drop the outliers from the DataFrame
    df_cleaned = df.drop(indices_to_remove)
    return df_cleaned


def format_data(data):
    df = data.copy()
    # Manipulate dividentRate column
    if "dividendRate" in df.columns:
        df["dividendRate"] = df["dividendRate"].fillna(0)
    # Manipulate dividentYield columns
    if "dividendYield" in df.columns:
        df["dividendYield"] = df["dividendYield"].fillna(0)
    # Manipulate marketCap columns
    if "marketCap" in df.columns:
        df.sort_values(by=["marketCap"], ascending=False, inplace=True)
        million_cols = ["marketCap"]
        df[million_cols] = df[million_cols].div(1_000_000)
    # Create fiftyTwoWeekRange column
    if ("fiftyTwoWeekLow" in df.columns) and ("fiftyTwoWeekHigh" in df.columns):
        fiftyTwoWeekRange = (
            (df["currentPrice"] - df["fiftyTwoWeekLow"])
            / (df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"])
        ) * 100
        fiftyTwoWeekRange_index = df.columns.tolist().index("fiftyTwoWeekHigh") + 1
        df.insert(fiftyTwoWeekRange_index, "fiftyTwoWeekRange", fiftyTwoWeekRange)
    # Assign Column Categories
    percentage_cols = [
        "52WeekChange",
        # "grossMargins",
        # "operatingMargins",
        # "ebitdaMargins",
        # "profitMargins",
    ]
    numerical_cols = [
        x for x in df.select_dtypes(include=[float]).columns if x not in percentage_cols
    ]

    # Fix Styling
    def make_pretty(styler):
        style_dict = {}
        style_dict.update({c: "{:.2f}" for c in numerical_cols})
        style_dict.update({c: "{:.2f}%" for c in percentage_cols})
        style_dict.update({c: "{:,.0f} M" for c in million_cols})
        styler.format(style_dict)
        styler.set_properties(**{"border": "0.1px solid black"})
        styler.hide(axis="index")
        styler.set_properties(
            subset=[
                "symbol",
                "shortName",
                "longName",
                "exchange",
                "quoteType",
                "sectorDisp",
            ],
            **{"text-align": "left"},
        )
        # Set the bar visualization
        if "fiftyTwoWeekRange" in df.columns:
            styler.bar(
                subset=["fiftyTwoWeekRange"],
                align="mid",
                color=["salmon", "cornflowerblue"],
            )
        if "marketCap" in df.columns:
            styler.bar(
                subset=["marketCap"],
                align="mid",
                color=["salmon", "cornflowerblue"],
            )
        gradient_cols = ["52WeekChange", "priceToBook"] + percentage_cols
        # Set background gradients
        for gc in gradient_cols:
            styler.background_gradient(subset=[gc], cmap="Greens")
        return styler

    return (
        df.style.pipe(make_pretty)
        .set_caption("Fundamental Indicators")
        .set_table_styles(
            [
                {"selector": "th.col_heading", "props": "text-align: center"},
                {
                    "selector": "caption",
                    "props": [
                        ("text-align", "center"),
                        ("font-size", "11pt"),
                        ("font-weight", "bold"),
                    ],
                },
            ]
        )
    )


"""
# ==============================================================
# Process
# ==============================================================
"""

TIMETAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOTSYMBOL = "DAX"
SAMPLESYMBOLS = 0
SEED = 1990


if __name__ == "__main__":
    # Get list of symbols
    # markets = get_markets()
    # stocksymbols = get_symbols(market="GR")
    stocksymbols = get_symbols(market=ROOTSYMBOL)
    print(f"Symbols for {ROOTSYMBOL}: {len(stocksymbols)}")

    # Sample symbols
    if SAMPLESYMBOLS > 0:
        np.random.seed(SEED)
        symbols = np.random.choice(stocksymbols, SAMPLESYMBOLS, replace=False).tolist()
        OUTPUTPATH = (
            f"data/screener_{ROOTSYMBOL}_{str(SAMPLESYMBOLS)}_{TIMETAG}.html".lower()
        )
        print(f"Symbols for {ROOTSYMBOL} afer sampling: {len(stocksymbols)}")
    else:
        symbols = stocksymbols
        OUTPUTPATH = f"data/screener_{ROOTSYMBOL}_{TIMETAG}.html".lower()

    # fetch data for the symbols list
    fdata = pd.DataFrame([get_fundamentals_yf(sym) for sym in tqdm(symbols)])
    print(f"Downloaded fundamentals for: {fdata.shape[0]} symbols")

    # Drop Rows based on missingness
    threshold_pct = 0.49
    subsetcols = [x for x in fdata.columns if not x in ["quoteType", "symbol"]]
    threshold = int(len(subsetcols) * threshold_pct)
    odata = fdata.dropna(subset=subsetcols, thresh=threshold).copy()
    print(f"Dropped {len(fdata) - len(odata)} rows due to missingness")

    # Remove outliers
    data = remove_outliers(odata, iqr_factor=3)
    print(f"Dropped {len(odata) - len(data)} rows due to outliers")

    # Create html output
    format_data(data).to_html(OUTPUTPATH)
    print(f'Output saved to "{OUTPUTPATH}"')
