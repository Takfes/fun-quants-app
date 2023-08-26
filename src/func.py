import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, norm, percentileofscore, skew


def convert_date(x):
    return datetime.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")


def find_below_threshold_missingness(
    data: pd.DataFrame, threshold: float = 0.0
) -> List:
    return (
        (data.isnull().sum() / data.shape[0])
        .loc[lambda x: x <= threshold]
        .index.tolist()
    )


def rebalance_weights(weights: np.ndarray, threshold: float = 0.0001) -> np.ndarray:
    weights[weights < threshold] = 0
    total_weight = np.sum(weights)
    if total_weight < 0:
        weights = weights / total_weight
    return weights


def annual_risk_return(
    data: pd.DataFrame, risk_free_rate: float = 0.017
) -> pd.DataFrame:
    pl_data = pl.from_pandas(data)
    returns = pl_data.select(
        [(pl.col(column).mean().alias(column)) for column in pl_data.columns]
    )
    risk = pl_data.select(
        [(pl.col(column).std().alias(column)) for column in pl_data.columns]
    )
    pd_returns = returns.to_pandas().T
    pd_risk = risk.to_pandas().T
    summary = pd.DataFrame(
        {
            "Returns": pd_returns.sum(axis=1) * 252,
            "Risk": pd_risk.sum(axis=1) * np.sqrt(252),
        }
    )
    summary["Sharpe"] = (summary.Returns - risk_free_rate) / summary.Risk
    return summary


def plot_assets_density(df):
    # Number of assets
    n_assets = df.shape[1]
    # Create a grid of plots
    fig, axes = plt.subplots(
        nrows=(n_assets + 1) // 2, ncols=2, figsize=(14, 2.5 * n_assets)
    )
    # Flatten the axes for easy iteration
    axes = axes.flatten()
    # Define the standard deviations and their corresponding probabilities
    stds = [1, 2, 3]
    probs = [0.68, 0.95, 0.997]
    outprobs = [0.16, 0.025, 0.0015]
    for i, col in enumerate(df.columns):
        # Calculate the first four moments
        mu = df[col].mean()
        sigma = df[col].std()
        skewness = skew(df[col])
        kurt = kurtosis(df[col])
        # Plot the density of the returns
        sns.kdeplot(df[col], ax=axes[i], shade=True, label="Observed Returns")
        # Plot the normal distribution for comparison
        x = np.linspace(df[col].min(), df[col].max(), 100)
        y = norm.pdf(x, mu, sigma)
        axes[i].plot(x, y, "r--", label="Normal Distribution")
        # Add vertical barriers and annotations
        for std, prob, outprob in zip(stds, probs, outprobs):
            # Theoretical barriers
            left_theoretical = mu - std * sigma
            right_theoretical = mu + std * sigma
            axes[i].axvline(left_theoretical, color="blue", linestyle="--", alpha=0.5)
            axes[i].axvline(right_theoretical, color="blue", linestyle="--", alpha=0.5)
            # Observed barriers
            left_observed = np.percentile(df[col], (1 - prob) / 2 * 100)
            right_observed = np.percentile(df[col], (1 + prob) / 2 * 100)
            axes[i].axvline(left_observed, color="green", linestyle="-.", alpha=0.5)
            axes[i].axvline(right_observed, color="green", linestyle="-.", alpha=0.5)
            # Annotations
            theoretical_pct = prob
            theoretical_pct_out = outprob
            observed_pct_left = percentileofscore(df[col], left_theoretical) / 100
            observed_pct_right = 1 - percentileofscore(df[col], right_theoretical) / 100
            axes[i].annotate(
                f"{theoretical_pct*100:.1f}% or {theoretical_pct_out:.2%} outside",
                (left_theoretical, 0),
                textcoords="offset points",
                # xytext=(-10, -10),
                xytext=(0, 60),
                ha="center",
                fontsize=8,
                color="blue",  # Color for theoretical
                rotation=25,
            )
            axes[i].annotate(
                f"{observed_pct_left*100:.1f}% outside",
                (left_theoretical, 0),
                textcoords="offset points",
                # xytext=(-10, -25),
                xytext=(0, 30),
                ha="center",
                fontsize=8,
                color="green",  # Color for observed
                rotation=25,
            )
            axes[i].annotate(
                f"{theoretical_pct*100:.1f}% or {theoretical_pct_out:.2%} outside",
                (right_theoretical, 0),
                textcoords="offset points",
                # xytext=(10, -10),
                xytext=(20, 60),
                ha="center",
                fontsize=8,
                color="blue",  # Color for theoretical
                rotation=25,
            )
            axes[i].annotate(
                f"{observed_pct_right*100:.1f}% outside",
                (right_theoretical, 0),
                textcoords="offset points",
                # xytext=(10, -25),
                xytext=(20, 30),
                ha="center",
                fontsize=8,
                color="green",  # Color for observed
                rotation=25,
            )
        # Set the title and subtitle
        title = f"Asset: {col}"
        subtitle = f"Mean: {mu:.2f}, Std: {sigma:.2f}, Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}"
        axes[i].set_title(title)
        axes[i].set_xlabel(subtitle)
        axes[i].legend()
    # Remove any unused subplots
    if n_assets % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


def plot_portfolios(
    assets_data: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    optimization_data: pd.DataFrame = None,
    annotate: bool = False,
    size: int = 15,
):
    # TODO vmin & vmax should be calculated based on the data
    # calculate max sharpe ratio portfolio
    max_sharpe_idx = portfolio_data.Sharpe.idxmax()
    max_sharpe = portfolio_data.loc[max_sharpe_idx]
    plt.figure(figsize=(15, 8))
    # plot generated portfolios
    plt.scatter(
        x=portfolio_data.loc[:, "Risk"],
        y=portfolio_data.loc[:, "Returns"],
        c=portfolio_data.loc[:, "Sharpe"],
        cmap="coolwarm",
        s=15,
        vmin=0.5,
        vmax=1.00,
        alpha=0.8,
    )
    plt.colorbar()
    # plot original assets used to generate portfolios
    plt.scatter(
        x=assets_data.loc[:, "Risk"],
        y=assets_data.loc[:, "Returns"],
        c=assets_data.loc[:, "Sharpe"],
        cmap="coolwarm",
        s=60,
        vmin=0.5,
        vmax=1.00,
        alpha=0.8,
        marker="D",
    )
    if annotate:
        for i in assets_data.index:
            plt.annotate(
                i,
                xy=(
                    assets_data.loc[assets_data.index == i, "Risk"].squeeze(),
                    assets_data.loc[assets_data.index == i, "Returns"].squeeze(),
                ),
                size=size,
            )
    # plot max sharpe ratio portfolio
    plt.scatter(
        x=max_sharpe["Risk"],
        y=max_sharpe["Returns"],
        c="orange",
        s=400,
        marker="*",
    )
    # plot optimization portfolio
    if not optimization_data is None:
        plt.scatter(
            x=optimization_data["Risk"],
            y=optimization_data["Returns"],
            c="red",
            s=400,
            marker="X",
        )
    plt.xlabel("Ann Risk (std)", fontsize=15)
    plt.ylabel("Ann Returns", fontsize=15)
    plt.title("Risk/Return/Sharpe Ratio", fontsize=20)
    plt.show()
