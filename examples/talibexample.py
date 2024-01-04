import talib
import yfinance as yf

data = yf.download("NVDA", start="2023-01-01", end="2023-12-28")
flag = talib.CDLMORNINGSTAR(data["Open"], data["High"], data["Low"], data["Close"])

list(talib.get_function_groups().keys())
candle_names = talib.get_function_groups()["Pattern Recognition"]

for candle in candle_names:
    data[candle] = getattr(talib, candle)(
        data["Open"], data["High"], data["Low"], data["Close"]
    )

data.loc[:, "Adj Close":].corr()
data.loc[:, "Adj Close":].corr().to_clipboard()
corr = data.loc[:, "Adj Close":].corr()["Adj Close"].sort_values(ascending=False)
top5_signals = corr.head(6).index.tolist()
data.loc[:, top5_signals].to_clipboard()
