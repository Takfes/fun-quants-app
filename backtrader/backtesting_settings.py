import numpy as np

strategy_settings_dictionary = {
    "ma": {"fast": [15, 25], "slow": 30},
    "dic": {
        "wma_period": 300,
        "rsi_period": 14,
        "rsi_value_long": 57,
        "rsi_value_short": 57,
        "stoploss": 0.03,  # [0.025,0.03,0.04]
        "takeprofit": 0.03,  # [0.025,0.03,0.04]
        "trstop": 0,
        "trstop_percent": 0.005,
        "emergency_exit": 1,
        "short_positions": 1,
        "period": 110,
        "factor": 0.618,
        "multiplier": 3,
        "printlog": False,
        "datasize": 50000,
    },
    "3h": {
        "stoploss": 0.03,
        "takeprofit": 0.03,
        "trstop": 0,
        "trstop_percent": 0.005,
        "short_positions": 1,
        "factor": 6.5,
        "atr_period": 170,
        "pivot_period": 100,
        "printlog": False,
        "datasize": 50000,
    },
}

strategy_analyzers = ["drawdown", "returns", "tradeanalyzer"]
