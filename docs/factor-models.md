## Resources

- [Complete Python and Machine Learning in Financial Analysis](https://www.udemy.com/course/python-and-machine-learning-in-financial-analysis/learn/lecture/28160152#overview)
- [Python for Financial Analysis and Algorithmic Trading](https://www.udemy.com/course/python-for-finance-and-trading-algorithms/learn/lecture/7613127#overview)
- [Kenneth French Factors - Dartmouth](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## Factor Models

- Capital Asset Pricing Model (CAPM)
- Fama-French Three-Factor Model
- Carhart Four-Factor Model (Extension of Fama-French)
- Fama-French Five-Factor Model
- APT (Arbitrage Pricing Theory)

### Columns

**Mkt_RF (Market Risk Premium)**: The excess return on the market portfolio over the risk-free rate. It's calculated as the return of a broad market index (like the S&P 500) minus the risk-free rate.

**SMB (Small Minus Big)**: The return on a portfolio of small stocks minus the return on a portfolio of big stocks. It's one of the factors used to explain performance anomalies in the market, representing the additional return investors expect from investing in small-cap stocks.

**HML (High Minus Low)**: The return on a portfolio of high book-to-market stocks minus the return on a portfolio of low book-to-market stocks. This factor captures the historical tendency for companies with high book-to-market ratios (value companies) to outperform those with low ratios (growth companies).

**RMW (Robust Minus Weak)**: The return on a portfolio of stocks with high operating profitability minus the return on a portfolio of stocks with low operating profitability. This factor is part of the Fama-French Five-Factor model.

**MOM (Momentum)**: This factor represents the tendency for stocks that have performed well in the past to continue performing well in the near future, and vice versa for stocks that have performed poorly. You can calculate this by looking at the past 12-month returns, excluding the most recent month, but it's often more reliable to use a pre-calculated series from a reputable source.
You can find the MOM data on the same Kenneth French Dartmouth website where you found the other factors. Look for datasets labeled "Momentum Factor" or similar.

**CMA (Conservative Minus Aggressive)**: The return on a portfolio of low investment firms minus the return on a portfolio of high investment firms. This factor is also part of the Fama-French Five-Factor model and represents the tendency for less aggressive, more conservative firms to yield higher returns.

**RF (Risk-Free Rate)**: The return on a virtually risk-free investment over the period. Often, short-term government treasury bills are used as the risk-free rate.
