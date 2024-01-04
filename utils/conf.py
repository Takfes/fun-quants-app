RISK_FREE_RATE = 0.017

metadata = [
    "shortName",  # A shorter or abbreviated name for the company.
    "longName",  # The full name of the company.
    "exchange",  # The stock exchange where the stock is traded.
    "quoteType",  # Indicates the type of financial instrument, e.g., stock, option, mutual fund.
    "sector",  # The industry or category of a company.
    # -----------------------------------------------
    "currentPrice",  # The current price of the stock.
    "targetLowPrice",  # The lowest analyst target price for the stock.
    "targetMeanPrice",  # The mean analyst target price for the stock.
    "targetMedianPrice",  # The median analyst target price for the stock.
    "targetHighPrice",  # The highest analyst target price for the stock.
    # -----------------------------------------------
    "recommendationMean",  # The mean recommendation of analysts for the stock.
    "recommendationKey",  # The consensus recommendation of analysts for the stock.
    "numberOfAnalystOpinions",  # The number of analysts that have issued recommendations for the stock.
    # -----------------------------------------------
    "marketCap",  # The total market value of a company's outstanding shares of stock. It's calculated by multiplying the stock's price by the total number of outstanding shares.
    "averageVolume",  # The average number of shares traded over a specific period, typically 30 days.
    "52WeekChange",  # The percentage change in a stock's price over the past 52 weeks.
    "earningsGrowth",  # The percentage growth in a company's earnings over a specific period.
    "revenueGrowth",  # The percentage growth in a company's revenue over a specific period.
    "revenuePerShare",  # The total revenue divided by the number of outstanding shares.
    "beta",  # A measure of a stock's volatility in relation to the overall market. A beta greater than 1 indicates higher volatility, while less than 1 indicates lower volatility.
    # -----------------------------------------------
    "priceToBook",  # The ratio of a company's stock price to its book value per share.
    "debtToEquity",  # The ratio of a company's total debt to its total equity.
    "returnOnEquity",  # A measure of a company's profitability relative to its shareholders' equity.
    "returnOnAssets",  # A measure of a company's profitability relative to its total assets.
    "profitMargins",  # The percentage of revenue that exceeds the cost of goods sold (profit).
    "ebitdaMargins",  # The percentage of total sales revenue that remains after deducting all operating expenses except interest, taxes, depreciation, and amortization.
    "grossMargins",  # The percentage of total sales revenue that the company retains after incurring the direct costs associated with producing goods/services.
    "operatingMargins",  # The percentage of total sales revenue that remains after deducting operating expenses.
    "enterpriseToEbitda",  # The ratio of a company's enterprise value to its earnings before interest, taxes, depreciation, and amortization (EBITDA).
    "enterpriseToRevenue",  # The ratio of a company's enterprise value (market cap plus debt minus cash) to its revenue.
    "currentRatio",  # A measure of a company's ability to cover its short-term liabilities with its short-term assets.
    "payoutRatio",  # The proportion of earnings paid out as dividends to shareholders, typically expressed as a percentage.
    "quickRatio",  # A measure of a company's ability to cover its short-term liabilities with its most liquid assets.
    "priceToSalesTrailing12Months",  # The ratio of a company's stock price to its revenue over the past 12 months.
    # -----------------------------------------------
    "dividendYield",  # The annual dividend payment divided by the stock's current market price. It indicates the income generated from an investment in the stock.
    "fiveYearAvgDividendYield",  # The average dividend yield over the past five years.
    # -----------------------------------------------
    "trailingPE",  # Price-to-Earnings ratio based on the past 12 months of earnings.
    "forwardPE",  # Price-to-Earnings ratio based on forecasted earnings for the next 12 months.
    "trailingEps",  # The sum of a company's earnings per share for the trailing 12 months.
    "forwardEps",  # The sum of a company's earnings per share for the next 12 months.
    "pegRatio",  # The ratio of a stock's price-to-earnings (P/E) ratio to the growth rate of its earnings for a specified time period.
    "trailingPegRatio",  # A stock's price-to-earnings ratio divided by the growth rate of its earnings for a specified time period.
    # -----------------------------------------------
    "overallRisk",  # The overall risk of a company.
    "auditRisk",  # The risk of an auditor issuing an incorrect opinion on a company's financial statements.
    "boardRisk",  # The risk of a company's board of directors not acting in the best interest of shareholders.
    "compensationRisk",  # The risk of a company's compensation policies and practices not being effective in incentivizing its management.
    "shareHolderRightsRisk",  # The risk of a company's shareholder rights not being adequately protected.
    # -----------------------------------------------
]
