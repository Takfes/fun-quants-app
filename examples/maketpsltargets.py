import pandas as pd


def calculate_tp_sl_days(close_prices, tp, sl):
    """
    For each closing price, calculate if T/P or S/L is hit first and after how many days.

    :param close_prices: A pandas Series of closing prices.
    :param tp: Target profit level as a positive float (e.g., 0.05 for 5%).
    :param sl: Stop loss level as a negative float (e.g., -0.05 for -5%).
    :return: A DataFrame with columns 'Hit' (T/P or S/L), 'Wait' days to hit and Returns
    """
    # Calculate daily returns
    returns = close_prices.pct_change()

    # Create a DataFrame to hold results
    results = pd.DataFrame(index=close_prices.index, columns=["Hit", "Wait", "Returns"])

    # Loop over the close prices
    for i in range(len(close_prices)):
        # Calculate the cumulative return from the current day forward
        forward_returns = (1 + returns.iloc[i:]).cumprod() - 1

        # Check if T/P or S/L is hit first
        tp_hit_days = forward_returns[forward_returns >= tp].index.min()
        sl_hit_days = forward_returns[forward_returns <= sl].index.min()

        # Determine which comes first, T/P or S/L
        if pd.isnull(tp_hit_days) and pd.isnull(sl_hit_days):
            # Neither T/P nor S/L is hit
            results.at[close_prices.index[i], "Hit"] = None
            results.at[close_prices.index[i], "Wait"] = None
            results.at[close_prices.index[i], "Returns"] = None
        elif pd.isnull(sl_hit_days) or (
            not pd.isnull(tp_hit_days) and tp_hit_days < sl_hit_days
        ):
            # T/P is hit first
            results.at[close_prices.index[i], "Hit"] = "TP"
            results.at[close_prices.index[i], "Wait"] = (
                tp_hit_days - close_prices.index[i]
            ).days
            results.at[close_prices.index[i], "Returns"] = forward_returns.loc[
                tp_hit_days
            ]
        else:
            # S/L is hit first
            results.at[close_prices.index[i], "Hit"] = "SL"
            results.at[close_prices.index[i], "Wait"] = (
                sl_hit_days - close_prices.index[i]
            ).days
            results.at[close_prices.index[i], "Returns"] = forward_returns.loc[
                sl_hit_days
            ]

    return results.dropna().astype(
        {"Hit": "str", "Wait": "int32", "Returns": "float64"}
    )


TICKER = "TSLA"
TP = 0.05
SL = -0.05

df = pd.DataFrame().ta.ticker(TICKER, period="5y", interval="1d")

df.plot(y="Close", figsize=(10, 5))
results = calculate_tp_sl_days(df["Close"], 0.05, -0.05)

results.Hit.value_counts()
results.Hit.value_counts(normalize=True)

results.Returns.hist(bins=100)
results.Returns.describe()

print(results)
