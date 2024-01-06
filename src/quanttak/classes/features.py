import pandas as pd
import talib


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

    def generate_momentum_indicators(self):
        open = self.open
        high = self.high
        low = self.low
        close = self.close
        volume = self.volume
        timeperiod = self.timeperiod
        slowperiod = self.slowperiod
        fastperiod = self.fastperiod
        # ADX - Average Directional Movement Index
        self.dict["MOM_ADX"] = talib.ADX(high, low, close, timeperiod=timeperiod)
        # ADXR - Average Directional Movement Index Rating
        self.dict["MOM_ADXR"] = talib.ADXR(high, low, close, timeperiod=timeperiod)
        # APO - Absolute Price Oscillator
        self.dict["MOM_APO"] = talib.APO(
            close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0
        )
        # AROON - Aroon
        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=timeperiod)
        self.dict["MOM_AROONDOWN"] = aroon_down
        self.dict["MOM_AROONUP"] = aroon_up
        # AROONOSC - Aroon Oscillator
        self.dict["MOM_AROONOSC"] = talib.AROONOSC(high, low, timeperiod=timeperiod)
        # BOP - Balance Of Power
        self.dict["MOM_BOP"] = talib.BOP(open, high, low, close)
        # CCI - Commodity Channel Index
        self.dict["MOM_CCI"] = talib.CCI(high, low, close, timeperiod=timeperiod)
        # CMO - Chande Momentum Oscillator
        self.dict["MOM_CMO"] = talib.CMO(close, timeperiod=timeperiod)
        # DX - Directional Movement Index
        self.dict["MOM_DX"] = talib.DX(high, low, close, timeperiod=timeperiod)
        # MACD - Moving Average Convergence/Divergence
        macd, macdsignal, macdhist = talib.MACD(
            close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=9
        )
        self.dict["MOM_MACD"] = macd
        self.dict["MOM_MACDSIGNAL"] = macdsignal
        self.dict["MOM_MACDHIST"] = macdhist
        # MFI - Money Flow Index
        self.dict["MOM_MFI"] = talib.MFI(
            high, low, close, volume, timeperiod=timeperiod
        )
        # MINUS_DI - Minus Directional Indicator
        self.dict["MOM_MINUSDI"] = talib.MINUS_DI(
            high, low, close, timeperiod=timeperiod
        )
        # MINUS_DM - Minus Directional Movement
        self.dict["MOM_MINUSDM"] = talib.MINUS_DM(high, low, timeperiod=timeperiod)
        # MOM - Momentum
        self.dict["MOM_MOM"] = talib.MOM(close, timeperiod=timeperiod)
        # PLUS_DI - Plus Directional Indicator
        self.dict["MOM_PLUSDI"] = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
        # PLUS_DM - Plus Directional Movement
        self.dict["MOM_PLUSDM"] = talib.PLUS_DM(high, low, timeperiod=timeperiod)
        # PPO - Percentage Price Oscillator
        self.dict["MOM_PPO"] = talib.PPO(
            close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0
        )
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        self.dict["MOM_ROC"] = talib.ROC(close, timeperiod=timeperiod)
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        self.dict["MOM_ROCP"] = talib.ROCP(close, timeperiod=timeperiod)
        # ROCR - Rate of change ratio: (price/prevPrice)
        self.dict["MOM_ROCR"] = talib.ROCR(close, timeperiod=timeperiod)
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        self.dict["MOM_ROCR100"] = talib.ROCR100(close, timeperiod=timeperiod)
        # RSI - Relative Strength Index
        self.dict["MOM_RSI"] = talib.RSI(close, timeperiod=timeperiod)
        # STOCH - Stochastic
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
        self.dict["MOM_STOCHK"] = slowk
        self.dict["MOM_STOCHD"] = slowd
        # STOCHF - Stochastic Fast
        fastk, fastd = talib.STOCHF(
            high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        self.dict["MOM_STOCHFK"] = fastk
        self.dict["MOM_STOCHFD"] = fastd
        # STOCHRSI - Stochastic Relative Strength Index
        fastk, fastd = talib.STOCHRSI(
            close, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        self.dict["MOM_TRIX"] = talib.TRIX(close, timeperiod=30)
        # ULTOSC - Ultimate Oscillator
        self.dict["MOM_ULTOSC"] = talib.ULTOSC(
            high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
        )
        # WILLR - Williams' %R
        self.dict["MOM_WILLR"] = talib.WILLR(high, low, close, timeperiod=timeperiod)

    def generate_volume_indicators(self):
        high = self.high
        low = self.low
        close = self.close
        volume = self.volume
        fastperiod = self.fastperiod
        slowperiod = self.slowperiod
        self.dict["VOLUME_AD"] = talib.AD(high, low, close, volume)
        self.dict["VOLUME_ADOSC"] = talib.ADOSC(
            high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod
        )
        self.dict["VOLUME_OBV"] = talib.OBV(close, volume)

    def generate_volatility_indicators(self):
        high = self.high
        low = self.low
        close = self.close
        timeperiod = self.timeperiod
        self.dict["VOLATILITY_ATR"] = talib.ATR(high, low, close, timeperiod=timeperiod)
        self.dict["VOLATILITY_NATR"] = talib.NATR(
            high, low, close, timeperiod=timeperiod
        )
        self.dict["VOLATILITY_TRANGE"] = talib.TRANGE(high, low, close)

    def generate_pattern_recognition(self):
        open = self.open
        high = self.high
        low = self.low
        close = self.close
        candle_names = talib.get_function_groups()["Pattern Recognition"]
        for candle in candle_names:
            self.dict["PATTERN_" + candle] = getattr(talib, candle)(
                open,
                high,
                low,
                close,
            )

    def generate_overlap_studies(self):
        high = self.high
        low = self.low
        close = self.close
        timeperiod = self.timeperiod
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0
        )
        self.dict["OVRP_BBANDS_UPPER"] = upper
        self.dict["OVRP_BBANDS_MIDDLE"] = middle
        self.dict["OVRP_BBANDS_LOWER"] = lower
        self.dict["OVRP_DEMA"] = talib.DEMA(close, timeperiod=timeperiod)
        self.dict["OVRP_EMA"] = talib.EMA(close, timeperiod=timeperiod)
        self.dict["OVRP_HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
        self.dict["OVRP_KAMA"] = talib.KAMA(close, timeperiod=timeperiod)
        self.dict["OVRP_MA"] = talib.MA(close, timeperiod=timeperiod, matype=0)
        # mama, fama = talib.MAMA(close, fastlimit=0, slowlimit=0)
        # self.dict["OVRP_MAMA"] = mama
        # self.dict["OVRP_FAMA"] = fama
        self.dict["OVRP_MIDPOINT"] = talib.MIDPOINT(close, timeperiod=timeperiod)
        self.dict["OVRP_MIDPRICE"] = talib.MIDPRICE(high, low, timeperiod=timeperiod)
        self.dict["OVRP_SAR"] = talib.SAR(high, low, acceleration=0, maximum=0)
        self.dict["OVRP_SMA"] = talib.SMA(close, timeperiod=timeperiod)
        self.dict["OVRP_T3"] = talib.T3(close, timeperiod=timeperiod, vfactor=0)
        self.dict["OVRP_TEMA"] = talib.TEMA(close, timeperiod=timeperiod)
        self.dict["OVRP_TRIMA"] = talib.TRIMA(close, timeperiod=timeperiod)
        self.dict["OVRP_WMA"] = talib.WMA(close, timeperiod=timeperiod)

    def generate_cycle_indicators(self):
        self.close
        self.dict["CYCLE_DCPERIOD"] = talib.HT_DCPERIOD(
            self.close
        )  # Hilbert Transform - Dominant Cycle Period
        self.dict["CYCLE_DCPHASE"] = talib.HT_DCPHASE(
            self.close
        )  # Hilbert Transform - Dominant Cycle Phase
        (
            self.dict["CYCLE_PHASOR_INPHASE"],
            self.dict["CYCLE_PHASOR_QUADRATURE"],
        ) = talib.HT_PHASOR(
            self.close
        )  # Hilbert Transform - Phasor Components
        (
            self.dict["CYCLE_SINE"],
            self.dict["CYCLE_LEADSINE"],
        ) = talib.HT_SINE(
            self.close
        )  # Hilbert Transform - SineWave
        self.dict["CYCLE_TRENDMODE"] = talib.HT_TRENDMODE(
            self.close
        )  # Hilbert Transform - Trend vs Cycle Mode

    def generate_statistic_functions(self):
        high = self.high
        low = self.low
        close = self.close
        timeperiod = self.timeperiod
        self.dict["STAT_BETA"] = talib.BETA(high, low, timeperiod=timeperiod)
        self.dict["STAT_CORREL"] = talib.CORREL(high, low, timeperiod=timeperiod)
        self.dict["STAT_LINEARREG"] = talib.LINEARREG(close, timeperiod=timeperiod)
        self.dict["STAT_LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(
            close, timeperiod=timeperiod
        )
        self.dict["STAT_LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(
            close, timeperiod=timeperiod
        )
        self.dict["STAT_LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(
            close, timeperiod=timeperiod
        )
        self.dict["STAT_STDDEV"] = talib.STDDEV(close, timeperiod=timeperiod, nbdev=1)
        self.dict["STAT_TSF"] = talib.TSF(close, timeperiod=timeperiod)
        self.dict["STAT_VAR"] = talib.VAR(close, timeperiod=timeperiod, nbdev=1)

    def generate_all_indicators(self):
        self.generate_momentum_indicators()
        self.generate_volume_indicators()
        self.generate_volatility_indicators()
        self.generate_pattern_recognition()
        self.generate_overlap_studies()
        self.generate_cycle_indicators()
        self.generate_statistic_functions()
        indicators = pd.DataFrame(self.dict, index=self.data.index)
        data = self.data.join(indicators)
        if self.dropna:
            shape_before = data.shape
            data.dropna(inplace=True)
            shape_after = data.shape
            print(f"Shape before dropping nas: {shape_before} and after: {shape_after}")
            print(f"Dropped {shape_before[0] - shape_after[0]} rows")
        return data
