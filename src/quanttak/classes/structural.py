import io
import zipfile
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf


@dataclass
class FMRegressor:
    x_years_ago = 5
    today = datetime.today()
    start_year_x_years_ago = datetime(today.year - x_years_ago, 1, 1)
    start_date = today.strftime("%Y-%m-%d")
    end_data = start_year_x_years_ago.strftime("%Y-%m-%d")

    def __init__(self, asset, start_date=start_date, end_date=end_data):
        self.asset: str = asset
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.model_CAPM = None
        self.model_FF3 = None
        self.model_Carhart = None
        self.model_FF5 = None

    def __repr__(self):
        return f"FactorModel(asset={self.asset}, start_date={self.start_date}, end_date={self.end_date})"

    def __str__(self):
        return f"FactorModel(asset={self.asset}, start_date={self.start_date}, end_date={self.end_date})"

    def get_asset_data(self):
        return yf.download(
            self.asset,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            group_by="columns",
        )

    @staticmethod
    def get_factor_data():
        def get_ff5data():
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            response = requests.get(url)
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            csv_file = zip_file.open(zip_file.namelist()[0])
            df = pd.read_csv(csv_file, skiprows=2)
            df = df.rename(columns={"Unnamed: 0": "Date"})
            df.columns = [x.replace("-", "_") for x in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df.set_index("Date", inplace=True)
            return df

        def get_momdata():
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
            response = requests.get(url)
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            csv_file = zip_file.open(zip_file.namelist()[0])
            df = pd.read_csv(csv_file, skiprows=13)
            df.columns = ["Date", "MOM"]
            df = df.dropna(subset=["MOM"])
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df.set_index("Date", inplace=True)
            return df

        ff5data = get_ff5data()
        momdata = get_momdata()
        data = ff5data.join(momdata, how="inner", on="Date")
        return data

    def report_sm_model(self, model):
        results = {
            "Intercept": model.params[0],
            "P-Value of Intercept": model.pvalues[0],
            "R-squared": model.rsquared,
            "Adjusted R-squared": model.rsquared_adj,
        }
        for idx, (param, pvalue) in enumerate(
            zip(model.params[1:], model.pvalues[1:]), 1
        ):
            results[f"Factor {idx} Coefficient"] = param
            results[f"Factor {idx} P-Value"] = pvalue
        return results

    def process_data(self):
        print(f"Retrieving asset data...")
        asset = self.get_asset_data()
        asset["Returns"] = asset["Close"].pct_change(1)
        asset.dropna(inplace=True)
        print(f"Retrieving factor data...")
        factors = self.get_factor_data()
        merged = asset[["Returns"]].join(factors, how="inner", rsuffix="_ff5")
        merged["Excess_Returns"] = merged["Returns"] - merged["RF"]
        return merged

    def fit(self):
        data = self.process_data()
        # Define independent variables for each model
        X_CAPM = data[["Mkt_RF"]]
        X_FF3 = data[["Mkt_RF", "SMB", "HML"]]
        X_Carhart = data[["Mkt_RF", "SMB", "HML", "MOM"]]
        X_FF5 = data[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
        y = data["Excess_Returns"]
        # Add a constant to the independent variable sets
        X_CAPM = sm.add_constant(X_CAPM)
        X_FF3 = sm.add_constant(X_FF3)
        X_Carhart = sm.add_constant(X_Carhart)
        X_FF5 = sm.add_constant(X_FF5)
        # Run the regressions
        self.model_CAPM = sm.OLS(y, X_CAPM).fit()
        self.model_FF3 = sm.OLS(y, X_FF3).fit()
        self.model_Carhart = sm.OLS(y, X_Carhart).fit()
        self.model_FF5 = sm.OLS(y, X_FF5).fit()
        self.results = {
            "CAPM": self.report_sm_model(self.model_CAPM),
            "FF3": self.report_sm_model(self.model_FF3),
            "Carhart": self.report_sm_model(self.model_Carhart),
            "FF5": self.report_sm_model(self.model_FF5),
        }
