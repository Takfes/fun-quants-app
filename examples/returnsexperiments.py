import pandas as pd
import ta
import talib
import yfinance as yf

pd.options.mode.chained_assignment = None  # Disable the SettingWithCopyWarning

from utils.func import get_fundamentals, get_symbols

START = "2015-01-01"
END = "2024-01-02"

# Get Symbols
symbols = get_symbols(index="SPX")
symbolssample = symbols[::50]

# Get Fundamentals Data
fundamentals = [get_fundamentals(symbol) for symbol in symbolssample]
fdata = pd.DataFrame(fundamentals)

# Get OHLC Data
yraw = yf.download(symbolssample, start=START, end=END, group_by="ticker")

ydata = (
    yraw.stack(level=0)
    .reset_index()
    .rename(columns={"level_1": "Symbol"})
    .sort_values(by=["Symbol", "Date"])
    .set_index("Date")
)
print(f"{ydata.shape=}")

# # Examine Best Returns frame
# data = []
# for x in range(30):
#     ydata[f"Returns{str(x)}d"] = ydata.groupby("Symbol")["Adj Close"].pct_change(x)
#     temp = ydata.groupby("Symbol")[f"Returns{str(x)}d"].describe() * 100
#     temp["Days"] = x
#     data.append(temp)
# foo = pd.concat(data).reset_index()
# foo.to_clipboard()

# # Log Returns
# ydata["LogReturns1d"] = (
#     ydata.groupby("Symbol")["Adj Close"].apply(np.log).diff(1).values
# )
# ydata["LogReturns3d"] = (
#     ydata.groupby("Symbol")["Adj Close"].apply(np.log).diff(3).values
# )
# ydata["LogReturns5d"] = (
#     ydata.groupby("Symbol")["Adj Close"].apply(np.log).diff(5).values
# )

# Returns
ydata["Returns1d"] = ydata.groupby("Symbol")["Adj Close"].pct_change(1)
ydata["Returns3d"] = ydata.groupby("Symbol")["Adj Close"].pct_change(3)
ydata["Returns5d"] = ydata.groupby("Symbol")["Adj Close"].pct_change(5)
ydata.dropna(inplace=True)
print(f"{ydata.shape=}")

# # Describe Returns
# ydata.groupby("Symbol")["Returns1d"].describe() * 100
# ydata.groupby("Symbol")["Returns3d"].describe() * 100
# ydata.groupby("Symbol")["Returns5d"].describe() * 100

# # Examine Data
# ydata.query("Symbol == 'DXC'").plot(y="Adj Close", kind="line")
# ydata.query("Symbol == 'DXC'").plot(y="Returns1d", kind="kernel", bins=100)
# ydata.query("Symbol == 'DXC'").to_clipboard()

# Add TA Features
grouped = ydata.groupby("Symbol")
tafdata = pd.concat(
    [
        ta.add_all_ta_features(
            grouped.get_group(x),
            open="Open",
            high="High",
            low="Low",
            close="Adj Close",
            volume="Volume",
            fillna=False,
        )
        for x in grouped.groups.keys()
    ]
)
tafdata.shape
print(f"{tafdata.shape=}")

# Add Candle Patterns
candle_names = talib.get_function_groups()["Pattern Recognition"]
grpd = tafdata.groupby("Symbol")

data = []
for gr in grpd.groups.keys():
    temp = grpd.get_group(gr)
    for candle in candle_names:
        temp[candle] = getattr(talib, candle)(
            temp["Open"], temp["High"], temp["Low"], temp["Close"]
        )
    data.append(temp)

cdldata = pd.concat(data)
print(f"{cdldata.shape=}")
cdldata.isnull().sum().sum()
