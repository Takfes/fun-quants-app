import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import pandas as pd
from backtesting_settings import strategy_analyzers, strategy_settings_dictionary
from strategies.BuyDip import BuyDip
from strategies.BuyHold import BuyHold
from strategies.Dictum import Dictum
from strategies.GoldenCross import GoldenCross
from strategies.TripleH import TripleH

import backtrader as bt
from backtrader import Cerebro
from helpers import parse_cerebro

strategies = {
    "ma": GoldenCross,
    "bnh": BuyHold,
    "dip": BuyDip,
    "dic": Dictum,
    "3h": TripleH,
}

optimizer = False
optreturn = True

# type = 'futures1'
# symbol = 'BTCUSDT'
# strategy = '3h'
# cash = 100
# risk = 0.025
# datasize = 100


def parse_user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        help="what kind of asset : ['stock','crypto','crypto15','futures15','futures1']",
        type=str,
    )
    parser.add_argument("symbol", help="which symbol to test", type=str)
    parser.add_argument(
        "strategy", help=f"which strategy to test : {list(strategies.keys())}", type=str
    )
    parser.add_argument("cash", help="set cash amount", type=str)
    parser.add_argument("risk", help="percentage of budget to risk", type=float)
    args = parser.parse_args()
    return args


def print_arguments(args):
    print(f"\n====================================\n")
    print(f"Command Line Arguments :")
    for k, v in args._get_kwargs():
        print(f"* {k} : {v}")
    print(f"\n====================================\n")


def get_price_series(type, symbol, con, datasize):
    if type == "stock":
        sql_string = (
            f"SELECT * FROM stockdaily WHERE symbol = '{symbol}' ORDER BY datetime"
        )
        price_series = (
            pd.read_sql(sql_string, con)
            .assign(datetime=lambda x: pd.to_datetime(x.datetime))
            .set_index("datetime")
        )
    elif type == "crypto":
        sql_string = (
            f"SELECT * FROM cryptodaily WHERE symbol = '{symbol}' ORDER BY datetime"
        )
        price_series = (
            pd.read_sql(sql_string, con)
            .assign(datetime=lambda x: pd.to_datetime(x.datetime))
            .set_index("datetime")
        )
    elif type == "crypto15":
        sql_string = f"SELECT * FROM crypto WHERE symbol = '{symbol}' ORDER BY datetime"
        price_series = (
            pd.read_sql(sql_string, con)
            .assign(datetime=lambda x: pd.to_datetime(x.datetime))
            .set_index("datetime")
        )
    elif type == "futures15":
        sql_string = f"SELECT * FROM futures15 WHERE symbol = '{symbol}' and openTime >= '2021-08-01' and openTime < '2021-08-13 15:30:00' ORDER BY openTimets"
        price_series = (
            pd.read_sql(sql_string, con)
            .assign(openTime=lambda x: pd.to_datetime(x.openTime))
            .set_index("openTime")
        )
    elif type == "futures1":
        # sql_string = f"SELECT * FROM futures1  WHERE symbol = '{symbol}' and openTime >= '2020-09-01' and openTime < '2021-08-31 15:30:00' ORDER BY openTimets"
        sql_string = f"SELECT * FROM futures1  WHERE symbol = '{symbol}' ORDER BY openTimets DESC limit {datasize}"
        price_series = (
            pd.read_sql(sql_string, con)
            .assign(openTime=lambda x: pd.to_datetime(x.openTime))
            .sort_values(by=["openTime"])
            .set_index("openTime")
        )
    return price_series


if __name__ == "__main__":
    start = time.time()

    args = parse_user_input()
    print_arguments(args)

    if args.strategy not in strategies.keys():
        print(f"Invalid strategy. Must be one of {list(strategies.keys())}")
        sys.exit()
    else:
        # strategy_settings = strategy_settings_dictionary['dic']
        # strategy_settings = strategy_settings_dictionary['3h']
        strategy_settings = strategy_settings_dictionary[args.strategy]
        datasize = strategy_settings["datasize"]
        # check whether any of the parameters passed is list
        # if so, enable cerebro.optstrategy instead of cerebro.addstrategy
        if any([isinstance(p, list) for p in strategy_settings.values()]):
            optimizer = True
            print(f"OPTIMIZER IS NOW OPEN")

    try:
        DATABASE_PATH = Path(config.DB_DIRECTORY) / config.DB_NAME
        con = sqlite3.connect(DATABASE_PATH)
        # price_series = get_price_series(type,symbol,con,datasize)
        price_series = get_price_series(args.type, args.symbol, con, datasize)
        print(
            f"> backtesting.py : fetched latest {price_series.shape[0]} rows for {args.symbol}"
        )
        print(
            f"> backtesting.py : period from {price_series.index.min()} to {price_series.index.max()}"
        )
        print(f"\n====================================\n")
    except Exception as e:
        print(f"DB CONNECTION ERROR")
        print(e)

    try:
        if not isinstance(price_series, pd.DataFrame):
            print("Expected Dataframe input ; EXITING ...")
            sys.exit()
        else:
            if price_series.empty:
                print("Received empty Dataframe ; EXITING ...")
                sys.exit()
            else:
                # initiate cerebro
                cerebro = bt.Cerebro()
                # cerebro.broker.setcash(cash)
                cerebro.broker.setcash(int(args.cash))
                start_portfolio_value = cerebro.broker.getvalue()

                # Add Dataset(s)
                feed = bt.feeds.PandasData(dataname=price_series)
                cerebro.adddata(feed)
                cerebro.broker.setcommission(commission=0.00, leverage=1)

                # Add Resample(s)
                if args.type == "futures1":
                    if args.strategy == "dic":
                        cerebro.resampledata(
                            feed, timeframe=bt.TimeFrame.Minutes, compression=15
                        )
                        cerebro.resampledata(
                            feed, timeframe=bt.TimeFrame.Minutes, compression=60
                        )
                    elif args.strategy == "3h":
                        cerebro.resampledata(
                            feed, timeframe=bt.TimeFrame.Minutes, compression=15
                        )

                # Add Strategy or Optimizer according to parameter input
                if not optimizer:
                    if args.strategy == "ma":
                        cerebro.addstrategy(
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            fast=strategy_settings.get("fast"),
                            slow=strategy_settings.get("slow"),
                        )
                    elif args.strategy == "dic":
                        cerebro.addstrategy(
                            # strategies['dic'],
                            # symbol = symbol,
                            # risk = risk,
                            # cash = cash,
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            wma_period=strategy_settings.get("wma_period"),
                            rsi_period=strategy_settings.get("rsi_period"),
                            rsi_value_long=strategy_settings.get("rsi_value_long"),
                            rsi_value_short=strategy_settings.get("rsi_value_short"),
                            stoploss=strategy_settings.get("stoploss"),
                            takeprofit=strategy_settings.get("takeprofit"),
                            trstop=strategy_settings.get("trstop"),
                            trstop_percent=strategy_settings.get("trstop_percent"),
                            short_positions=strategy_settings.get("short_positions"),
                            emergency_exit=strategy_settings.get("emergency_exit"),
                            period=strategy_settings.get("period"),
                            factor=strategy_settings.get("factor"),
                            multiplier=strategy_settings.get("multiplier"),
                            printlog=strategy_settings.get("printlog"),
                        )
                        # cerebro.addstrategy(
                        #     strategies['dic'],**strategy_settings)

                    elif args.strategy == "3h":
                        cerebro.addstrategy(
                            # strategies['3h'],
                            # symbol = symbol,
                            # risk = risk,
                            # cash = cash,
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            stoploss=strategy_settings.get("stoploss"),
                            takeprofit=strategy_settings.get("takeprofit"),
                            trstop=strategy_settings.get("trstop"),
                            trstop_percent=strategy_settings.get("trstop_percent"),
                            short_positions=strategy_settings.get("short_positions"),
                            factor=strategy_settings.get("factor"),
                            atr_period=strategy_settings.get("atr_period"),
                            pivot_period=strategy_settings.get("pivot_period"),
                            printlog=strategy_settings.get("printlog"),
                        )

                else:
                    if args.strategy == "ma":
                        cerebro.optstrategy(
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            fast=strategy_settings.get("fast"),
                            slow=strategy_settings.get("slow"),
                        )

                    elif args.strategy == "dic":
                        cerebro.optstrategy(
                            # strategies['dic'],
                            # symbol = symbol,
                            # risk = risk,
                            # cash = cash,
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            wma_period=strategy_settings.get("wma_period"),
                            rsi_period=strategy_settings.get("rsi_period"),
                            rsi_value_long=strategy_settings.get("rsi_value_long"),
                            rsi_value_short=strategy_settings.get("rsi_value_short"),
                            stoploss=strategy_settings.get("stoploss"),
                            takeprofit=strategy_settings.get("takeprofit"),
                            short_positions=strategy_settings.get("short_positions"),
                            trstop=strategy_settings.get("trstop"),
                            trstop_percent=strategy_settings.get("trstop_percent"),
                            emergency_exit=strategy_settings.get("emergency_exit"),
                            period=strategy_settings.get("period"),
                            factor=strategy_settings.get("factor"),
                            multiplier=strategy_settings.get("multiplier"),
                            printlog=strategy_settings.get("printlog"),
                        )

                    elif args.strategy == "3h":
                        cerebro.optstrategy(
                            # strategies['3h'],
                            # symbol = symbol,
                            # risk = risk,
                            # cash = cash,
                            strategies[args.strategy],
                            symbol=args.symbol,
                            risk=args.risk,
                            cash=args.cash,
                            stoploss=strategy_settings.get("stoploss"),
                            takeprofit=strategy_settings.get("takeprofit"),
                            short_positions=strategy_settings.get("short_positions"),
                            trstop=strategy_settings.get("trstop"),
                            trstop_percent=strategy_settings.get("trstop_percent"),
                            factor=strategy_settings.get("factor"),
                            atr_period=strategy_settings.get("atr_period"),
                            pivot_period=strategy_settings.get("pivot_period"),
                            printlog=strategy_settings.get("printlog"),
                        )

                # Add Analyzer
                if "drawdown" in strategy_analyzers:
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
                if "sharpe" in strategy_analyzers:
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
                if "returns" in strategy_analyzers:
                    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
                if "periodstats" in strategy_analyzers:
                    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="periodstats")
                if "tradeanalyzer" in strategy_analyzers:
                    cerebro.addanalyzer(
                        bt.analyzers.TradeAnalyzer, _name="tradeanalyzer"
                    )

                # Cerebro Results
                # Optimizer Results
                if optimizer:
                    # w/ optreturn
                    if optreturn:
                        R = cerebro.run(stdstats=False)
                        print(f">>> Cerebro finished {len(R)} trials !!! <<<")
                        dfr = parse_cerebro(R, strategy=args.strategy).sort_values(
                            by=["td_pnl_gross_total"], ascending=False
                        )
                        dfr.insert(0, "strategy", args.strategy)
                        timetag = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dfr.to_csv(
                            f"./cerebro_results/opt_{args.strategy}_{timetag}_{args.symbol}.csv",
                            index=False,
                        )

                    # w/o optreturn
                    else:
                        R = cerebro.run(stdstats=False, optreturn=False)
                        print(f">>> Cerebro finished {len(R)} trials !!! <<<")
                        # dfr = parse_cerebro(R,strategy = strategy).sort_values(by=['td_pnl_gross_total'],ascending=False)
                        dfr = parse_cerebro(R, strategy=args.strategy).sort_values(
                            by=["td_pnl_gross_total"], ascending=False
                        )
                        dfr.insert(0, "strategy", args.strategy)
                        timetag = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dfr.to_csv(
                            f"./cerebro_results/opt_{args.strategy}_{timetag}_{args.symbol}.csv",
                            index=False,
                        )

                # Results w/o optimizer
                else:
                    R = cerebro.run()
                    # dfr = parse_cerebro(R,strategy='3h').sort_values(by=['td_pnl_gross_total'],ascending=False)
                    dfr = parse_cerebro(R, strategy=args.strategy).sort_values(
                        by=["td_pnl_gross_total"], ascending=False
                    )
                    dfr.insert(0, "strategy", args.strategy)
                    timetag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dfr.to_csv(
                        f"./cerebro_results/noopt_{args.strategy}_{timetag}_{args.symbol}.csv",
                        index=False,
                    )

                    end_portfolio_value = cerebro.broker.getvalue()
                    pnl = end_portfolio_value - start_portfolio_value
                    print(f"\nStarting Portfolio Value: {start_portfolio_value:.2f}")
                    print(f"Final Portfolio Value: {end_portfolio_value:.2f}")
                    print(f"PnL: {pnl:.2f} - {(pnl/end_portfolio_value)*100:.2f}%")

                    # Plot Results
                    # cerebro.plot()

    except Exception as e:
        print("Error in cerebro section ; EXITING ...")
        print(e)
        # sys.exit()

    end = time.time()
    print(f"Total execution time {end-start}")
