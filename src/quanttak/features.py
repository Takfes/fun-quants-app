import pandas as pd
import talib

TIMEPERIOD = 14
FASTPERIOD = 12
SLOWPERIOD = 26


def generate_momentum_indicators(
    open,
    high,
    low,
    close,
    volume,
    timeperiod: int = TIMEPERIOD,
    fastperiod: int = FASTPERIOD,
    slowperiod: int = SLOWPERIOD,
    return_dataframe: bool = True,
):
    datadict = {}
    datadict["MOM_ADX"] = talib.ADX(high, low, close, timeperiod=timeperiod)
    datadict["MOM_ADXR"] = talib.ADXR(high, low, close, timeperiod=timeperiod)
    datadict["MOM_APO"] = talib.APO(
        close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0
    )
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=timeperiod)
    datadict["MOM_AROONDOWN"] = aroon_down
    datadict["MOM_AROONUP"] = aroon_up
    datadict["MOM_AROONOSC"] = talib.AROONOSC(high, low, timeperiod=timeperiod)
    datadict["MOM_BOP"] = talib.BOP(open, high, low, close)
    datadict["MOM_CCI"] = talib.CCI(high, low, close, timeperiod=timeperiod)
    datadict["MOM_CMO"] = talib.CMO(close, timeperiod=timeperiod)
    datadict["MOM_DX"] = talib.DX(high, low, close, timeperiod=timeperiod)
    macd, macdsignal, macdhist = talib.MACD(
        close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=9
    )
    datadict["MOM_MACD"] = macd
    datadict["MOM_MACDSIGNAL"] = macdsignal
    datadict["MOM_MACDHIST"] = macdhist
    datadict["MOM_MFI"] = talib.MFI(high, low, close, volume, timeperiod=timeperiod)
    datadict["MOM_MINUSDI"] = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)
    datadict["MOM_MINUSDM"] = talib.MINUS_DM(high, low, timeperiod=timeperiod)
    datadict["MOM_MOM"] = talib.MOM(close, timeperiod=timeperiod)
    datadict["MOM_PLUSDI"] = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
    datadict["MOM_PLUSDM"] = talib.PLUS_DM(high, low, timeperiod=timeperiod)
    datadict["MOM_PPO"] = talib.PPO(
        close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0
    )
    datadict["MOM_ROC"] = talib.ROC(close, timeperiod=timeperiod)
    datadict["MOM_ROCP"] = talib.ROCP(close, timeperiod=timeperiod)
    datadict["MOM_ROCR"] = talib.ROCR(close, timeperiod=timeperiod)
    datadict["MOM_ROCR100"] = talib.ROCR100(close, timeperiod=timeperiod)
    datadict["MOM_RSI"] = talib.RSI(close, timeperiod=timeperiod)
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    datadict["MOM_STOCHK"] = slowk
    datadict["MOM_STOCHD"] = slowd
    fastk, fastd = talib.STOCHF(
        high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    datadict["MOM_STOCHFK"] = fastk
    datadict["MOM_STOCHFD"] = fastd
    fastk, fastd = talib.STOCHRSI(
        close, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    datadict["MOM_TRIX"] = talib.TRIX(close, timeperiod=30)
    datadict["MOM_ULTOSC"] = talib.ULTOSC(
        high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )
    datadict["MOM_WILLR"] = talib.WILLR(high, low, close, timeperiod=timeperiod)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_volume_indicators(
    high,
    low,
    close,
    volume,
    fastperiod: int = FASTPERIOD,
    slowperiod: int = SLOWPERIOD,
    return_dataframe=True,
):
    datadict = {}
    datadict["VOLUME_AD"] = talib.AD(high, low, close, volume)
    datadict["VOLUME_ADOSC"] = talib.ADOSC(
        high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod
    )
    datadict["VOLUME_OBV"] = talib.OBV(close, volume)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_volatility_indicators(
    high, low, close, timeperiod: int = TIMEPERIOD, return_dataframe=True
):
    datadict = {}
    datadict["VOLATILITY_ATR"] = talib.ATR(high, low, close, timeperiod=timeperiod)
    datadict["VOLATILITY_NATR"] = talib.NATR(high, low, close, timeperiod=timeperiod)
    datadict["VOLATILITY_TRANGE"] = talib.TRANGE(high, low, close)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_pattern_recognition(open, high, low, close, return_dataframe=True):
    datadict = {}
    candle_names = talib.get_function_groups()["Pattern Recognition"]
    for candle in candle_names:
        datadict["PATTERN_" + candle] = getattr(talib, candle)(
            open,
            high,
            low,
            close,
        )
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_overlap_studies(
    high, low, close, timeperiod: int = TIMEPERIOD, return_dataframe=True
):
    datadict = {}
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0
    )
    datadict["OVRP_BBANDS_UPPER"] = upper
    datadict["OVRP_BBANDS_MIDDLE"] = middle
    datadict["OVRP_BBANDS_LOWER"] = lower
    datadict["OVRP_DEMA"] = talib.DEMA(close, timeperiod=timeperiod)
    datadict["OVRP_EMA"] = talib.EMA(close, timeperiod=timeperiod)
    datadict["OVRP_HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
    datadict["OVRP_KAMA"] = talib.KAMA(close, timeperiod=timeperiod)
    datadict["OVRP_MA"] = talib.MA(close, timeperiod=timeperiod, matype=0)
    # mama, fama = talib.MAMA(close, fastlimit=0, slowlimit=0)
    # datadict["OVRP_MAMA"] = mama
    # datadict["OVRP_FAMA"] = fama
    datadict["OVRP_MIDPOINT"] = talib.MIDPOINT(close, timeperiod=timeperiod)
    datadict["OVRP_MIDPRICE"] = talib.MIDPRICE(high, low, timeperiod=timeperiod)
    datadict["OVRP_SAR"] = talib.SAR(high, low, acceleration=0, maximum=0)
    datadict["OVRP_SMA"] = talib.SMA(close, timeperiod=timeperiod)
    datadict["OVRP_T3"] = talib.T3(close, timeperiod=timeperiod, vfactor=0)
    datadict["OVRP_TEMA"] = talib.TEMA(close, timeperiod=timeperiod)
    datadict["OVRP_TRIMA"] = talib.TRIMA(close, timeperiod=timeperiod)
    datadict["OVRP_WMA"] = talib.WMA(close, timeperiod=timeperiod)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_cycle_indicators(close, return_dataframe=True):
    datadict = {}
    datadict["CYCLE_DCPERIOD"] = talib.HT_DCPERIOD(close)
    datadict["CYCLE_DCPHASE"] = talib.HT_DCPHASE(close)
    (
        datadict["CYCLE_PHASOR_INPHASE"],
        datadict["CYCLE_PHASOR_QUADRATURE"],
    ) = talib.HT_PHASOR(close)
    (
        datadict["CYCLE_SINE"],
        datadict["CYCLE_LEADSINE"],
    ) = talib.HT_SINE(close)
    datadict["CYCLE_TRENDMODE"] = talib.HT_TRENDMODE(close)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


def generate_statistic_functions(
    high, low, close, timeperiod: int = TIMEPERIOD, return_dataframe=True
):
    datadict = {}
    datadict["STAT_BETA"] = talib.BETA(high, low, timeperiod=timeperiod)
    datadict["STAT_CORREL"] = talib.CORREL(high, low, timeperiod=timeperiod)
    datadict["STAT_LINEARREG"] = talib.LINEARREG(close, timeperiod=timeperiod)
    datadict["STAT_LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(
        close, timeperiod=timeperiod
    )
    datadict["STAT_LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(
        close, timeperiod=timeperiod
    )
    datadict["STAT_LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(
        close, timeperiod=timeperiod
    )
    datadict["STAT_STDDEV"] = talib.STDDEV(close, timeperiod=timeperiod, nbdev=1)
    datadict["STAT_TSF"] = talib.TSF(close, timeperiod=timeperiod)
    datadict["STAT_VAR"] = talib.VAR(close, timeperiod=timeperiod, nbdev=1)
    if return_dataframe:
        return pd.DataFrame(datadict, index=close.index)
    else:
        return datadict


class FeatureEngineer:
    def __init__(
        self,
        data,
        colname_open: str = "Open",
        colname_high: str = "High",
        colname_low: str = "Low",
        colname_close: str = "Adj Close",
        colname_volume: str = "Volume",
        dropna: bool = True,
        timeperiod: int = 14,
        fastperiod: int = 12,
        slowperiod: int = 26,
    ):
        self.data = data
        self.colname_open = colname_open
        self.colname_high = colname_high
        self.colname_low = colname_low
        self.colname_close = colname_close
        self.colname_volume = colname_volume
        self.open = self.data[colname_open]
        self.high = self.data[colname_high]
        self.low = self.data[colname_low]
        self.close = self.data[colname_close]
        self.volume = self.data[colname_volume]
        self.timeperiod = timeperiod
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.dropna = dropna
        self.dict = {}

    def __str__(self):
        print("wrapper around talib : https://ta-lib.github.io/ta-lib-python/")
        return f"FeatureEngineer(data shape={self.data.shape}, open={self.colname_open}, high={self.colname_high}, low={self.colname_low}, close={self.colname_close}, volume={self.colname_volume}, timeperiod={self.timeperiod}, fastperiod={self.fastperiod}, slowperiod={self.slowperiod}, dropna={self.dropna})"

    def __repr__(self):
        return self.__str__()

    def generate_all_indicators(self):
        self.dict.update(
            generate_momentum_indicators(
                self.open,
                self.high,
                self.low,
                self.close,
                self.volume,
                self.timeperiod,
                self.fastperiod,
                self.slowperiod,
                return_dataframe=False,
            )
        )
        self.dict.update(
            generate_volume_indicators(
                self.high,
                self.low,
                self.close,
                self.volume,
                self.fastperiod,
                self.slowperiod,
                return_dataframe=False,
            )
        )
        self.dict.update(
            generate_volatility_indicators(
                self.high, self.low, self.close, self.timeperiod, return_dataframe=False
            )
        )
        self.dict.update(
            generate_pattern_recognition(
                self.open, self.high, self.low, self.close, return_dataframe=False
            )
        )
        self.dict.update(
            generate_overlap_studies(
                self.high, self.low, self.close, self.timeperiod, return_dataframe=False
            )
        )
        self.dict.update(generate_cycle_indicators(self.close, return_dataframe=False))
        self.dict.update(
            generate_statistic_functions(
                self.high, self.low, self.close, self.timeperiod, return_dataframe=False
            )
        )
        indicators = pd.DataFrame(self.dict, index=self.data.index)
        data = self.data.join(indicators)
        if self.dropna:
            shape_before = data.shape
            data.dropna(inplace=True)
            shape_after = data.shape
            print(f"Shape before dropping nas: {shape_before} and after: {shape_after}")
            print(f"Dropped {shape_before[0] - shape_after[0]} rows")
        return data
