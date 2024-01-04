import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

grsymbols = [
    "AAAK.AT",
    "AAAP.AT",
    "ADMIE.AT",
    "AEGN.AT",
    "AKRIT.AT",
    "ALMY.AT",
    "ALPHA.AT",
    "ANDRO.AT",
    "ANEK.AT",
    "ANEPO.AT",
    "ASCO.AT",
    "ASTAK.AT",
    "ATRUST.AT",
    "ATTICA.AT",
    "AVAX.AT",
    "AVE.AT",
    "BELA.AT",
    "BIOKA.AT",
    "BIOSK.AT",
    "BIOT.AT",
    "BRIQ.AT",
    "BYTE.AT",
    "CAIROMEZ.AT",
    "CENER.AT",
    "CENTR.AT",
    "CNLCAP.AT",
    "CPI.AT",
    "CPLPB1.AT",
    "DAIOS.AT",
    "DOMIK.AT",
    "DOPPLER.AT",
    "DROME.AT",
    "DUR.AT",
    "EEE.AT",
    "EKTER.AT",
    "ELBE.AT",
    "ELGEK.AT",
    "ELHA.AT",
    "ELHAB1.AT",
    "ELIN.AT",
    "ELLAKTOR.AT",
    "ELPE.AT",
    "ELSTR.AT",
    "ELTON.AT",
    "ENTER.AT",
    "EPIL.AT",
    "EPSIL.AT",
    "ETE.AT",
    "EUPIC.AT",
    "EUROB.AT",
    "EUROC.AT",
    "EVROF.AT",
    "EXAE.AT",
    "EYAPS.AT",
    "EYDAP.AT",
    "FIER.AT",
    "FLEXO.AT",
    "FOODL.AT",
    "FOYRK.AT",
    "FRIGO.AT",
    "GEBKA.AT",
    "GEKTERNA.AT",
    "GEKTERNAB3.AT",
    "HAIDE.AT",
    "HTO.AT",
    "IATR.AT",
    "IKTIN.AT",
    "ILYDA.AT",
    "INKAT.AT",
    "INLIF.AT",
    "INLOT.AT",
    "INTEK.AT",
    "INTERCO.AT",
    "INTET.AT",
    "INTRK.AT",
    "KAMP.AT",
    "KARE.AT",
    "KEKR.AT",
    "KEPEN.AT",
    "KLM.AT",
    "KMOL.AT",
    "KORDE.AT",
    "KRI.AT",
    "KTILA.AT",
    "KYLO.AT",
    "KYRI.AT",
    "LAMDA.AT",
    "LAMPS.AT",
    "LANAC.AT",
    "LAVI.AT",
    "LEBEK.AT",
    "LOGISMOS.AT",
    "LYK.AT",
    "MATHIO.AT",
    "MEDIC.AT",
    "MERKO.AT",
    "MEVA.AT",
    "MIG.AT",
    "MIN.AT",
    "MODA.AT",
    "MOH.AT",
    "MOTO.AT",
    "MOYZK.AT",
    "MPITR.AT",
    "MYTIL.AT",
    "NAKAS.AT",
    "NAYP.AT",
    "NIKAS.AT",
    "NOVALB1.AT",
    "OLTH.AT",
    "OLYMP.AT",
    "OPAP.AT",
    "OPTRON.AT",
    "OTOEL.AT",
    "PAIR.AT",
    "PAP.AT",
    "PERF.AT",
    "PETRO.AT",
    "PLAIS.AT",
    "PLAKR.AT",
    "PLAT.AT",
    "PPA.AT",
    "PPC.AT",
    "PRD.AT",
    "PREMIA.AT",
    "PRODEA.AT",
    "PROF.AT",
    "PROFK.AT",
    "PVMEZZ.AT",
    "QUAL.AT",
    "QUEST.AT",
    "REALCONS.AT",
    "REVOIL.AT",
    "SAR.AT",
    "SATOK.AT",
    "SIDMA.AT",
    "SPACE.AT",
    "TATT.AT",
    "TELL.AT",
    "TENERGY.AT",
    "TITC.AT",
    "TPEIR.AT",
    "TRASTOR.AT",
    "VARNH.AT",
    "VIO.AT",
    "VOSYS.AT",
    "XYLEK.AT",
    "XYLEP.AT",
    "YALCO.AT",
]

metadata = [
    "shortName",  # A shorter or abbreviated name for the company.
    "longName",  # The full name of the company.
    "exchange",  # The stock exchange where the stock is traded.
    "quoteType",  # Indicates the type of financial instrument, e.g., stock, option, mutual fund.
    "industry",
    "sector",  # The industry or category of a company.
    "currentPrice",  # The current price of the stock.
    "marketCap",  # The total market value of a company's outstanding shares of stock. It's calculated by multiplying the stock's price by the total number of outstanding shares.
    # "averageVolume",  # The average number of shares traded over a specific period, typically 30 days.
    # "forwardEps",
    # "forwardPE",
    # "pegRatio",
    # "freeCashflow",
    # "priceToBook",
    # "returnOnEquity",
    # "priceToSalesTrailing12Months",
    # "payoutRatio",
    # "dividendYield",
    # "beta",
    # "currentRatio",
    # "52WeekChange",  # The percentage change in a stock's price over the past 52 weeks.
    # "fiftyTwoWeekLow",
    # "fiftyTwoWeekHigh",
    # # ---------------------------------------------------------------------------
    # "targetLowPrice",  # The lowest analyst target price for the stock.
    # "targetMeanPrice",  # The mean analyst target price for the stock.
    # "targetMedianPrice",  # The median analyst target price for the stock.
    # "targetHighPrice",  # The highest analyst target price for the stock.
    # "recommendationMean",  # The mean recommendation of analysts for the stock.
    # "recommendationKey",  # The consensus recommendation of analysts for the stock.
    # "numberOfAnalystOpinions",  # The number of analysts that have issued recommendations for the stock.
    # "52WeekChange",  # The percentage change in a stock's price over the past 52 weeks.
    # "earningsGrowth",  # The percentage growth in a company's earnings over a specific period.
    # "revenueGrowth",  # The percentage growth in a company's revenue over a specific period.
    # "revenuePerShare",  # The total revenue divided by the number of outstanding shares.
    # "beta",  # A measure of a stock's volatility in relation to the overall market. A beta greater than 1 indicates higher volatility, while less than 1 indicates lower volatility.
    # "priceToBook",  # The ratio of a company's stock price to its book value per share.
    # "debtToEquity",  # The ratio of a company's total debt to its total equity.
    # "returnOnEquity",  # A measure of a company's profitability relative to its shareholders' equity.
    # "returnOnAssets",  # A measure of a company's profitability relative to its total assets.
    # "profitMargins",  # The percentage of revenue that exceeds the cost of goods sold (profit).
    # "ebitdaMargins",  # The percentage of total sales revenue that remains after deducting all operating expenses except interest, taxes, depreciation, and amortization.
    # "grossMargins",  # The percentage of total sales revenue that the company retains after incurring the direct costs associated with producing goods/services.
    # "operatingMargins",  # The percentage of total sales revenue that remains after deducting operating expenses.
    # "enterpriseToEbitda",  # The ratio of a company's enterprise value to its earnings before interest, taxes, depreciation, and amortization (EBITDA).
    # "enterpriseToRevenue",  # The ratio of a company's enterprise value (market cap plus debt minus cash) to its revenue.
    # "currentRatio",  # A measure of a company's ability to cover its short-term liabilities with its short-term assets.
    # "payoutRatio",  # The proportion of earnings paid out as dividends to shareholders, typically expressed as a percentage.
    # "quickRatio",  # A measure of a company's ability to cover its short-term liabilities with its most liquid assets.
    # "priceToSalesTrailing12Months",  # The ratio of a company's stock price to its revenue over the past 12 months.
    # "dividendYield",  # The annual dividend payment divided by the stock's current market price. It indicates the income generated from an investment in the stock.
    # "fiveYearAvgDividendYield",  # The average dividend yield over the past five years.
    # "trailingPE",  # Price-to-Earnings ratio based on the past 12 months of earnings.
    # "forwardPE",  # Price-to-Earnings ratio based on forecasted earnings for the next 12 months.
    # "trailingEps",  # The sum of a company's earnings per share for the trailing 12 months.
    # "forwardEps",  # The sum of a company's earnings per share for the next 12 months.
    # "pegRatio",  # The ratio of a stock's price-to-earnings (P/E) ratio to the growth rate of its earnings for a specified time period.
    # "trailingPegRatio",  # A stock's price-to-earnings ratio divided by the growth rate of its earnings for a specified time period.
    # "overallRisk",  # The overall risk of a company.
    # "auditRisk",  # The risk of an auditor issuing an incorrect opinion on a company's financial statements.
    # "boardRisk",  # The risk of a company's board of directors not acting in the best interest of shareholders.
    # "compensationRisk",  # The risk of a company's compensation policies and practices not being effective in incentivizing its management.
    # "shareHolderRightsRisk",  # The risk of a company's shareholder rights not being adequately protected.
]


def fetch_fundamentals(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.get_info()
    data = {}
    data[f"ticker"] = symbol
    for metric in metadata:
        data[f"{metric}"] = info.get(metric, None)
    return data


def make_pretty(styler):
    # Column formatting
    styler.format(
        {
            # "EPS (fwd)": "${:.2f}",
            # "P/E (fwd)": "{:.2f}",
            # "PEG": "{:.2f}",
            # "FCFY": "{:.2f}%",
            # "PB": "{:.2f}",
            # "ROE": "{:.2f}",
            # "P/S (trail)": "{:.2f}",
            # "DPR": "{:.2f}%",
            # "DY": "{:.2f}%",
            # "CR": "{:.2f}",
            # "Beta": "{:.2f}",
            # "52w Low": "${:.2f}",
            "currentPrice": "{:.2f}€",
            "marketCap": "€ {:.2f}€",
            # "52w High": "${:.2f}",
            # "52w Range": "{:.2f}%",
        }
    )
    # # Set the bar visualization
    # styler.bar(subset=["52w Range"], align="mid", color=["salmon", "cornflowerblue"])

    # Grid
    styler.set_properties(**{"border": "0.1px solid black"})

    # Set background gradients
    styler.background_gradient(subset=["currentPrice"], cmap="Greens")
    styler.background_gradient(subset=["marketCap"], cmap="Greens")
    # styler.background_gradient(subset=["EPS (fwd)"], cmap="Greens")
    # styler.background_gradient(subset=["P/E (fwd)"], cmap="Greens")
    # styler.background_gradient(subset=["PEG"], cmap="Greens")
    # styler.background_gradient(subset=["FCFY"], cmap="Greens")
    # styler.background_gradient(subset=["PB"], cmap="Greens")
    # styler.background_gradient(subset=["ROE"], cmap="Greens")
    # styler.background_gradient(subset=["P/S (trail)"], cmap="Greens")
    # styler.background_gradient(subset=["DPR"], cmap="Greens")
    # styler.background_gradient(subset=["DY"], cmap="Greens")
    # styler.background_gradient(subset=["CR"], cmap="Greens")

    # No index
    styler.hide(axis="index")

    # Tooltips
    styler.set_tooltips(
        ttips,
        css_class="tt-add",
        props=[
            ("visibility", "hidden"),
            ("position", "absolute"),
            ("background-color", "salmon"),
            ("color", "black"),
            ("z-index", 1),
            ("padding", "3px 3px"),
            ("margin", "2px"),
        ],
    )
    # Left text alignment for some columns
    styler.set_properties(
        # subset=["Symbol", "Name", "Industry"], **{"text-align": "left"}
        subset=["ticker", "shortName", "exchange", "quoteType", "industry", "sector"],
        **{"text-align": "left"},
    )
    return styler


def populate_tt(df, tt_data, col_name):
    stats = df[col_name].describe()

    per25 = round(stats.loc["25%"], 2)
    per50 = round(stats.loc["50%"], 2)
    per75 = round(stats.loc["75%"], 2)

    # Get position based on the column name
    pos = df.columns.to_list().index(col_name)

    for index, row in df.iterrows():
        pe = row[col_name]
        if pe == stats.loc["min"]:
            tt_data[index][pos] = "Lowest"
        elif pe == stats.loc["max"]:
            tt_data[index][pos] = "Hightest"
        elif pe <= per25:
            tt_data[index][pos] = "25% of companies under {}".format(per25)
        elif pe <= per50:
            tt_data[index][pos] = "50% of companies under {}".format(per50)
        elif pe <= per75:
            tt_data[index][pos] = "75% of companies under {}".format(per75)
        else:
            tt_data[index][pos] = "25% of companies over {}".format(per75)


data = []
for ticker in tqdm(grsymbols[::10]):
    try:
        temp = fetch_fundamentals(ticker)
        data.append(temp)
    except Exception:
        print(f"ERROR --> {ticker}")

df = pd.DataFrame(data)

# Initialize tool tip data - each column is set to '' for each row
tt_data = [["" for x in range(len(df.columns))] for y in range(len(df))]

# Gather tool tip data for indicators
populate_tt(df, tt_data, "currentPrice")
populate_tt(df, tt_data, "marketCap")

# Create a tool tip DF
ttips = pd.DataFrame(data=tt_data, columns=df.columns, index=df.index)

# Add table caption and styles to DF
df_styled = (
    df.style.pipe(make_pretty)
    .set_caption("Fundamental Indicators")
    .set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center"},
            {
                "selector": "caption",
                "props": [
                    ("text-align", "center"),
                    ("font-size", "11pt"),
                    ("font-weight", "bold"),
                ],
            },
        ]
    )
)

df_styled.to_html("data/fundamentals.html")

data = {
    "Symbol": [],
    "Name": [],
    "Industry": [],
    "EPS (fwd)": [],
    "P/E (fwd)": [],
    "PEG": [],
    "FCFY": [],
    "PB": [],
    "ROE": [],
    "P/S (trail)": [],
    "DPR": [],
    "DY": [],
    "CR": [],
    "Beta": [],
    "Price": [],
    "52w Low": [],
    "52w High": [],
}


def parse_yf_info(json_data):
    data["Symbol"].append(json_data["symbol"])
    data["Name"].append(json_data["longName"])
    data["Industry"].append(json_data["industry"])
    data["Price"].append(json_data["currentPrice"])

    if "forwardEps" in json_data:
        data["EPS (fwd)"].append(json_data["forwardEps"])
    else:
        data["EPS (fwd)"].append(np.nan)

    if "forwardPE" in json_data:
        data["P/E (fwd)"].append(json_data["forwardPE"])
    else:
        data["P/E (fwd)"].append(np.nan)

    if "pegRatio" in json_data:
        data["PEG"].append(json_data["pegRatio"])
    else:
        data["PEG"].append(np.nan)

    if ("freeCashflow" in json_data) and ("marketCap" in json_data):
        fcfy = (json_data["freeCashflow"] / json_data["marketCap"]) * 100
        data["FCFY"].append(round(fcfy, 2))
    else:
        data["FCFY"].append(np.nan)

    if "priceToBook" in json_data:
        data["PB"].append(json_data["priceToBook"])
    else:
        data["PB"].append(np.nan)

    if "returnOnEquity" in json_data:
        data["ROE"].append(json_data["returnOnEquity"])
    else:
        data["ROE"].append(np.nan)

    if "priceToSalesTrailing12Months" in json_data:
        data["P/S (trail)"].append(json_data["priceToSalesTrailing12Months"])
    else:
        data["P/S (trail)"].append(np.nan)

    if "priceToSalesTrailing12Months" in json_data:
        data["DPR"].append(json_data["payoutRatio"] * 100)
    else:
        data["DPR"].append(np.nan)

    if "dividendYield" in json_data:
        data["DY"].append(json_data["dividendYield"])
    else:
        data["DY"].append(0.0)

    if "beta" in json_data:
        data["Beta"].append(json_data["beta"])
    else:
        data["Beta"].append(np.nan)

    if "currentRatio" in json_data:
        data["CR"].append(json_data["currentRatio"])
    else:
        data["CR"].append(np.nan)

    if "fiftyTwoWeekLow" in json_data:
        data["52w Low"].append(json_data["fiftyTwoWeekLow"])
    else:
        data["52w Low"].append(np.nan)

    if "fiftyTwoWeekHigh" in json_data:
        data["52w High"].append(json_data["fiftyTwoWeekHigh"])
    else:
        data["52w High"].append(np.nan)


for ticker in tqdm(grsymbols[::10]):
    stock = yf.Ticker(ticker)
    try:
        json_data = stock.info
        parse_yf_info(json_data)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Ticker: {ticker}")

dataset = pd.DataFrame(data)
