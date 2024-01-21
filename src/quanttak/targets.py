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
    :return: A DataFrame with columns 'Hit' (T/P or S/L), 'Patience' days to hit and Returns
    """
    # Calculate daily returns
    returns = close_prices.pct_change()

    # Create a DataFrame to hold results
    results = pd.DataFrame(
        index=close_prices.index,
    )

    # Loop over the close prices
    for i in range(1, len(close_prices)):
        # ! check whether this is range(0,..) or range(1,..)
        # Calculate the cumulative return from the current day forward
        forward_returns = (1 + returns.iloc[i:]).cumprod() - 1

        # Check if T/P or S/L is hit first
        tp_hit_days = forward_returns[forward_returns >= tp].index.min()
        sl_hit_days = forward_returns[forward_returns <= sl].index.min()

        # Determine which comes first, T/P or S/L
        if pd.isnull(tp_hit_days) and pd.isnull(
            sl_hit_days
        ):  # * Neither T/P nor S/L is hit
            results.at[close_prices.index[i], "hit_date"] = None
            results.at[close_prices.index[i], "patience"] = None
            results.at[close_prices.index[i], "hit_price"] = None
            results.at[close_prices.index[i], "returns"] = None
            results.at[close_prices.index[i], "hit"] = None

        elif pd.isnull(sl_hit_days) or (
            not pd.isnull(tp_hit_days) and tp_hit_days < sl_hit_days
        ):  # * T/P is hit first
            results.at[close_prices.index[i], "hit_date"] = tp_hit_days
            results.at[close_prices.index[i], "patience"] = (
                close_prices.index.get_loc(tp_hit_days) + 1 - i
            )
            results.at[close_prices.index[i], "hit_price"] = close_prices.loc[
                tp_hit_days
            ]
            results.at[close_prices.index[i], "returns"] = (
                close_prices.loc[tp_hit_days] - close_prices.iloc[i - 1]
            ) / close_prices.iloc[
                i - 1
            ]  # ! check whether this is i or i-1
            results.at[close_prices.index[i], "hit"] = "TP"

        else:  # * S/L is hit first
            results.at[close_prices.index[i], "hit_date"] = sl_hit_days
            results.at[close_prices.index[i], "patience"] = (
                close_prices.index.get_loc(sl_hit_days) + 1 - i
            )
            results.at[close_prices.index[i], "hit_price"] = close_prices.loc[
                sl_hit_days
            ]
            results.at[close_prices.index[i], "returns"] = (
                close_prices.loc[sl_hit_days] - close_prices.iloc[i - 1]
            ) / close_prices.iloc[
                i - 1
            ]  # ! check whether this is i or i-1
            results.at[close_prices.index[i], "hit"] = "SL"

    return (
        results.dropna()
        .assign(hit_binary=lambda x: (x.hit == "TP").astype("int"))
        .join(close_prices, how="inner")
    )


def lookahead_downsampling(x_set, y_set, steer, fraction_to_sample, random_seed=None):
    """
    Downsample a training set based on a steering dataframe and a given sampling ratio.

    Parameters:
    train_set (pd.DataFrame): The training set to be downsampled.
    steer (pd.DataFrame): The 'target_tpsl_out' steering dataframe.
    fraction_to_sample (float): The fraction of the train set to sample.
    random_seed (int, optional): Random seed for reproducibility.

    Returns:
    pd.DataFrame: The downsampled training set.
    """
    # Joining the training set with the steer dataframe
    joined_df = steer.join(x_set, how="right")

    # Filtering rows where 'hit_date' is in the index
    filtered_df = joined_df[joined_df.hit_date.isin(joined_df.index)]

    # Calculate the reciprocal of the unique count of 'hit_date'
    filtered_df["unique_count"] = filtered_df.groupby("hit_date")["hit_date"].transform(
        "size"
    )
    filtered_df["reciprocal_unique_count"] = 1 / filtered_df["unique_count"]

    # Compute sampling probabilities
    filtered_df["sampling_prob"] = (
        filtered_df["reciprocal_unique_count"]
        / filtered_df["reciprocal_unique_count"].sum()
    )

    # Perform the sampling
    sampled_df = filtered_df.sample(
        frac=fraction_to_sample,
        replace=False,
        weights="sampling_prob",
        random_state=random_seed,
    )
    # Returning the sampled DataFrame with only the columns from the original training set
    return sampled_df[x_set.columns].copy(), y_set.loc[sampled_df.index].copy()
