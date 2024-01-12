import numpy as np
import pandas as pd


def make_target_rtns(
    close_prices: pd.Series,
    period: int = 1,
    return_type: str = "simple",
    binary: bool = False,
) -> pd.Series:
    """
    Calculate future return rates from close prices and optionally convert to binary target.
    Parameters:
        close_prices (pd.Series): Series of close prices.
        period (int): Number of periods to calculate return over.
        return_type (str): Type of return calculation ('log' or 'simple').
        binary (bool): If True, convert returns to binary (1 if positive, 0 otherwise).
    Returns:
        pd.Series: Calculated future returns or binary target.
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if return_type == "log":
        returns = np.log(close_prices.shift(-period)) - np.log(close_prices)
    elif return_type == "simple":
        returns = close_prices.shift(-period).pct_change(period)
    else:
        raise ValueError(f"Invalid return type '{return_type}'")
    returns = returns.dropna()
    returns.name = "returns"
    if binary:
        return (returns > 0).astype(int)
    else:
        return returns


def make_target_tpsl(close_prices, tp: float = 0.03, sl: float = -0.03):
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
    return (
        results.dropna()
        .astype({"Hit": "str", "Wait": "int32", "Returns": "float64"})
        .assign(HitBinary=lambda x: (x.Hit == "TP").astype("int"))
        .rename(columns=lambda x: x.lower())
    )
