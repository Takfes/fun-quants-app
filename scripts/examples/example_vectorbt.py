import vectorbt as vbt
import yfinance as yf

# Download historical data for a stock
START = "2020-01-01"
END = "2024-05-01"
tickers = "AMZN"
data = yf.download("AMZN", start=START, end=END)
prices = data["Close"]

# Calculate short-term and long-term moving averages
short_ma = prices.rolling(window=50).mean()
long_ma = prices.rolling(window=200).mean()

# Generate buy and sell signals
entries = short_ma > long_ma
exits = short_ma < long_ma

# Create a portfolio based on the signals
portfolio = vbt.Portfolio.from_signals(prices, entries, exits)

# Plot the portfolio value
portfolio.plot().show()

# Print performance statistics
print(portfolio.stats())
