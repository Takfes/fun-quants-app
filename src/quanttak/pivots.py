import matplotlib.pyplot as plt
import pandas as pd


def find_pivots(close_series, window=5):
    """
    Identify pivot points in the 'Close' time series data and characterize as minima or maxima.

    :param close_series: pandas Series of close prices.
    :param window: The number of days on either side to use for identifying local extrema.
    :return: pandas DataFrame with pivot points and their type (minima or maxima).
    """
    # Initialize pivot DataFrame
    pivots = pd.DataFrame(index=close_series.index, columns=["Pivot", "Type"])

    # Find local maxima and minima
    local_max = (
        close_series == close_series.rolling(window=2 * window + 1, center=True).max()
    )
    local_min = (
        close_series == close_series.rolling(window=2 * window + 1, center=True).min()
    )

    # Mark the identified pivots and their types
    pivots.loc[local_max, "Pivot"] = close_series[local_max]
    pivots.loc[local_max, "Type"] = "max"
    pivots.loc[local_min, "Pivot"] = close_series[local_min]
    pivots.loc[local_min, "Type"] = "min"

    return pivots.dropna()


def plot_pivots(close_series, window=5):
    """
    Plot the 'Close' time series and annotate local maxima in green and minima in red.

    :param close_series: pandas Series of close prices.
    :param window: The number of periods on either side to use for identifying local extrema.
    """
    # Identify pivots using the identify_pivots function
    pivots = identify_pivots(close_series, window)

    # Set up the plot
    plt.figure(figsize=(14, 7))
    plt.plot(close_series.index, close_series, label="Close Price", alpha=0.5)

    # Plot local maxima
    maxima = pivots[pivots["Type"] == "max"]
    plt.scatter(
        maxima.index,
        maxima["Pivot"],
        color="green",
        label="Local Maxima",
        marker="^",
        s=100,
    )

    # Plot local minima
    minima = pivots[pivots["Type"] == "min"]
    plt.scatter(
        minima.index,
        minima["Pivot"],
        color="red",
        label="Local Minima",
        marker="v",
        s=100,
    )

    # Show the plot with a legend
    plt.legend()
    plt.title("Close Prices with Pivot Points")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()


TICKER = "AAPL"

df = pd.DataFrame().ta.ticker(TICKER, period="5y", interval="1d")
# Example usage
pivot_points = identify_pivots(df["Close"], window=22)

# Merge pivot points with original DataFrame
df[["Pivots", "Type"]] = pivot_points

plot_pivots(df["Close"], window=10)
