import pandas as pd
import requests

pd.options.display.max_rows = 150

url = "https://scanner.tradingview.com/global/scan"

payload = {}
headers = {}

response = requests.request("POST", url, headers=headers, data=payload)

data = []
for item in response.json()["data"]:
    item_provider = item["s"].split(":")[0]
    item_detailed = item["s"].split(":")[1]
    data.append({"provider": item_provider, "detailed": item_detailed})

df = pd.DataFrame(data)

# df.detailed.nunique()
# df.groupby("provider").count().sort_values(by="detailed", ascending=False).to_clipboard()
# df.groupby("provider").count().sort_values(by="detailed", ascending=False).head(100)

stock_exchanges = [
    "AMEX",  # NYSE American, formerly known as the American Stock Exchange
    "ASX",  # Australian Securities Exchange
    "ATHEX",  # Athens Stock Exchange
    "BCBA",  # Bolsa de Comercio de Buenos Aires
    "BER",  # Berlin Stock Exchange
    "BET",  # Bucharest Stock Exchange
    "BIST",  # Borsa İstanbul
    "BM&FBOVESPA",  # B3 - Brasil Bolsa Balcão
    "BMV",  # Bolsa Mexicana de Valores (Mexican Stock Exchange)
    "BSE",  # Bombay Stock Exchange
    "BX",  # NASDAQ OMX BX
    "CBOT",  # Chicago Board of Trade
    "CSE",  # Colombo Stock Exchange
    "DSEBD",  # Dhaka Stock Exchange
    "EURONEXT",  # Pan-European stock exchange
    "FWB",  # Frankfurter Wertpapierbörse (Frankfurt Stock Exchange)
    "GPW",  # Warsaw Stock Exchange
    "HAM",  # Hamburg Stock Exchange
    "HKEX",  # Hong Kong Stock Exchange
    "HNX",  # Hanoi Stock Exchange
    "HOSE",  # Ho Chi Minh Stock Exchange
    "IDX",  # Indonesia Stock Exchange
    "JSE",  # Johannesburg Stock Exchange
    "KRX",  # Korea Exchange
    "LSE",  # London Stock Exchange
    "LUXSE",  # Luxembourg Stock Exchange
    "MEXC",  # Mexican Stock Exchange
    "MIL",  # Borsa Italiana (Milan Stock Exchange)
    "MOEX",  # Moscow Exchange
    "MUN",  # Munich Stock Exchange
    "MYX",  # Malaysia Exchange
    "NASDAQ",  # National Association of Securities Dealers Automated Quotations
    "NEO",  # NEO Exchange in Canada
    "NSE",  # National Stock Exchange of India
    "NYSE",  # New York Stock Exchange
    "OMXSTO",  # Nasdaq Stockholm
    "OSL",  # Oslo Stock Exchange
    "PSX",  # Pakistan Stock Exchange
    "SET",  # Stock Exchange of Thailand
    "SGX",  # Singapore Exchange
    "SIX",  # SIX Swiss Exchange
    "SSE",  # Shanghai Stock Exchange
    "SWB",  # Stuttgart Stock Exchange
    "SZSE",  # Shenzhen Stock Exchange
    "TADAWUL",  # Saudi Stock Exchange
    "TAIFEX",  # Taiwan Futures Exchange
    "TASE",  # Tel Aviv Stock Exchange
    "TPEX",  # Taipei Exchange
    "TRADEGATE",  # Tradegate Exchange
    "TSE",  # Tokyo Stock Exchange
    "TSX",  # Toronto Stock Exchange
    "TSXV",  # TSX Venture Exchange
    "TWSE",  # Taiwan Stock Exchange
    "VIE",  # Vienna Stock Exchange
    "XETR",  # Electronic trading system for the Frankfurt Stock Exchange
]

providers_of_interest = [
    "NYSE",
    "NASDAQ",
    "LSE",
    "BSE",
    "SSE",
    "TSE",
    "ATHEX",
]  # "EURONEXT",

# df[df.provider.isin(providers_of_interest)]

df[df.provider == "AMEX"].sort_values(by="detailed").sample(50)

df[df.detailed == "SPY"]
df[df.detailed == "IVV"]
df[df.detailed == "0JFF"]
df[df.detailed == "RUT"]

providers_of_interest_suffix = {
    "NYSE": "",
    "NASDAQ": "",
    "LSE": ".L",
    "BSE": ".BO",
    "SSE": ".SS",
    "TSE": ".T",
    "ATHEX": ".AT",
}

df_selection = df[df.provider.isin(providers_of_interest)].copy().reset_index(drop=True)
df_selection["suffix"] = df_selection.provider.map(providers_of_interest_suffix)
df_selection["ticker"] = df_selection.detailed + df_selection.suffix
df_selection.sort_values(by=["provider", "ticker"], inplace=True)

# df_selection[df_selection.provider == "NYSE"]

df_selection.to_excel("data/se_tickers.xlsx", index=False)
