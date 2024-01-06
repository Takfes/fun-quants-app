import backtrader as bt
import numpy as np

debug = False


# Custom Indicator
class DICK(bt.Indicator):
    lines = ("vwma", "bbt", "bbb")  # output lines (array)
    params = (("period", 110), ("factor", 0.618), ("multiplier", 3.0))

    def __init__(self):
        # indicate min period ; i.e. buffering period
        self.addminperiod(self.p.period)

    def next(self):
        # calculate vwma
        highp = np.array(self.data.high.get(size=self.p.period))
        lowp = np.array(self.data.low.get(size=self.p.period))
        closep = np.array(self.data.close.get(size=self.p.period))
        volumep = np.array(self.data.volume.get(size=self.p.period))
        hlcp = (highp + lowp + closep) / 3.0
        sumprodp = hlcp * volumep
        vwma = sum(sumprodp) / sum(volumep)
        # add vwma line
        self.lines.vwma[0] = vwma
        # calculate stdev hlc
        std = np.std(hlcp)
        # add bbt & bbb lines
        self.lines.bbt[0] = vwma + (self.p.multiplier * self.p.factor * std)
        self.lines.bbb[0] = vwma - (self.p.multiplier * self.p.factor * std)

        if debug:
            print("> highp : ", "\n", type(highp), "\n", highp, "\n")
            print("> lowp : ", "\n", type(lowp), "\n", lowp, "\n")
            print("> closep : ", "\n", type(closep), "\n", closep, "\n")
            print("> volumep : ", "\n", type(volumep), "\n", volumep, "\n")
            print("> hlcp : ", "\n", type(hlcp), "\n", hlcp, "\n")
            print("> sumprodp : ", "\n", type(sumprodp), "\n", sumprodp, "\n")
            print("> vwma : ", "\n", type(vwma), "\n", vwma, "\n")
            print("> std(hlcp) : ", "\n", type(std), "\n", std, "\n")


# Strategy


class Dictum(bt.Strategy):
    params = (
        ("symbol", "unknown"),
        ("cash", 1000),
        ("risk", 0.1),
        ("wma_period", 300),
        ("rsi_period", 14),
        ("rsi_value_long", 57),
        ("rsi_value_short", 57),
        ("stoploss", 0.01),
        ("takeprofit", 0.01),
        ("trstop", 0),
        ("trstop_percent", 0.005),
        ("short_positions", 0),
        ("emergency_exit", 1),
        ("period", 110),
        ("factor", 0.618),
        ("multiplier", 3.0),
        ("printlog", False),
    )

    def __init__(self):
        # settings
        # self.params.printlog = True
        self.signal_number = 1
        self.currency_format = 6
        self.starting_cash = self.broker.getvalue()
        self.accuracy_rate = 0
        self.total_signals = 0
        self.pnl = 0

        # 1 minute data
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataclose = self.datas[0].close

        # 15 minute data
        self.do = self.datas[1].open
        self.dh = self.datas[1].high
        self.dl = self.datas[1].low
        self.dc = self.datas[1].close

        # 60 minute data
        self.ho = self.datas[2].open
        self.hh = self.datas[2].high
        self.hl = self.datas[2].low
        self.hc = self.datas[2].close

        # indicators
        self.rsi = bt.indicators.RSI_SMA(self.datas[2], period=self.params.rsi_period)
        self.wma = bt.indicators.WeightedMovingAverage(
            self.datas[1], period=self.params.wma_period
        )
        dick = self.dick = DICK(
            self.datas[1],
            period=self.p.period,
            factor=self.p.factor,
            multiplier=self.p.multiplier,
        )
        dick.plotinfo.subplot = False

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def sizer(self):
        # TAKIS
        # amount_to_invest = self.broker.cash * self.params.risk
        # self.size = round((amount_to_invest / self.datas[0].close), 3)

        # PREKS
        amount_to_invest = self.starting_cash * (
            self.params.risk / self.params.stoploss
        )  # ALMOST FIXED AMOUNT OF LOSS
        # amount_to_invest = self.broker.cash * (self.params.risk / self.params.stoploss)  # AMOUNT OF LOSS CHANGES OVER TIME ACCORDING TO CURRENT CASH
        # amount_to_invest = self.broker.cash  # ENTER WITH 100% CASH ON EVERY SIGNAL
        self.currency_format = str(self.dc[0])[::-1].find(".")
        self.size = round((amount_to_invest / self.dc[0]), self.currency_format)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            # LONG POSITIONS
            if order.isbuy():
                self.executed_price = order.executed.price
                # IF THERE IS OPEN POSITION (just received open signal and position opened)
                if self.position:
                    self.entry_price = self.executed_price
                    self.log(f"(2) BUY EXECUTED : {self.entry_price:.2f}")
                # IF THERE IS NO OPEN POSITION (just received close signal and position closed)
                else:
                    # IF POSITION CLOSED WITH PROFIT
                    if self.profit_loss == "profit":
                        self.log(
                            f"(4) BUY EXECUTED : {order.executed.price:.2f}. PROFIT: {self.broker.getvalue() - self.wallet:.2f}"
                        )
                        self.log(f"(5) CURRENT WALLET : {self.broker.getvalue():.2f}")
                        self.accuracy_rate += 1
                        self.total_signals += 1
                        self.pnl += self.broker.getvalue() - self.wallet
                        self.log(
                            f"(6) ACCURACY RATE {self.accuracy_rate}/{self.total_signals} or {(self.accuracy_rate/self.total_signals)*100:.2f}%"
                        )
                    # IF POSITION CLOSED WITH LOSS
                    elif self.profit_loss == "loss":
                        self.log(
                            f"(4) BUY EXECUTED : {order.executed.price:.2f}. LOSS: {self.broker.getvalue() - self.wallet:.2f}"
                        )
                        self.log(f"(5) CURRENT WALLET : {self.broker.getvalue():.2f}")
                        self.total_signals += 1
                        self.pnl += self.broker.getvalue() - self.wallet
                        self.log(
                            f"(6) ACCURACY RATE {self.accuracy_rate}/{self.total_signals} or {(self.accuracy_rate/self.total_signals)*100:.2f}%"
                        )
            # SHORT POSITIONS
            elif order.issell():
                self.executed_price = order.executed.price
                # IF THERE IS OPEN POSITION (just received open signal and position opened)
                if self.position:
                    self.entry_price = order.executed.price
                    self.log(f"(2) SELL EXECUTED : {order.executed.price:.2f}")
                # IF THERE IS NO OPEN POSITION (just received close signal and position closed)
                else:
                    # IF POSITION CLOSED WITH PROFIT
                    if self.profit_loss == "profit":
                        self.log(
                            f"(4) SELL EXECUTED : {order.executed.price:.2f}. PROFIT: {self.broker.getvalue() - self.wallet:.2f}"
                        )
                        self.log(f"(5) CURRENT WALLET : {self.broker.getvalue():.2f}")
                        self.accuracy_rate += 1
                        self.total_signals += 1
                        self.pnl += self.broker.getvalue() - self.wallet
                        self.log(
                            f"(6) ACCURACY RATE {self.accuracy_rate}/{self.total_signals} or {(self.accuracy_rate/self.total_signals)*100:.2f}%"
                        )
                    # IF POSITION CLOSED WITH LOSS
                    elif self.profit_loss == "loss":
                        self.log(
                            f"(4) SELL EXECUTED : {order.executed.price:.2f}. LOSS: {self.broker.getvalue() - self.wallet:.2f}"
                        )
                        self.log(f"(5) CURRENT WALLET : {self.broker.getvalue():.2f}")
                        self.total_signals += 1
                        self.pnl += self.broker.getvalue() - self.wallet
                        self.log(
                            f"(6) ACCURACY RATE {self.accuracy_rate}/{self.total_signals} or {(self.accuracy_rate/self.total_signals)*100:.2f}%"
                        )
            self.bar_executed = len(self)
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(
            "(7) OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm)
        ) if trade.pnl > 0 else self.log(
            "(7) OPERATION LOSS, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm)
        )
        self.log(f'{50*"-"}\n')

    def next(self):
        # OPEN POSITIONS
        if not self.position:
            # OPEN LONG
            if (self.dc[0] > self.dick.lines.bbt) and (self.dc[0] > self.wma):
                self.log(f"SIGNAL NUMBER #{self.signal_number}")
                self.signal_number += 1
                # if (self.dataclose[0] > self.dick.lines.bbt) and (self.dataclose[0] > self.wma):
                self.sizer()
                self.log(
                    f"(1) SIGNAL NOTICE: Buy {self.size} shares of {self.params.symbol} at {self.data.close[0]:.2f}"
                )
                # self.buy(size=self.size)
                # self.currency_format = str(self.size)[::-1].find('.')
                self.buy(exectype=bt.Order.Market, size=self.size)
                if self.params.trstop == 1:
                    self.sell(
                        exectype=bt.Order.StopTrail,
                        size=self.size,
                        trailpercent=self.params.trstop_percent,
                    )
                self.wallet = self.broker.getvalue()

            # OPEN SHORT
            if self.params.short_positions:
                # if self.signal_short:
                if (self.dc[0] < self.dick.lines.bbb) and (self.dc[0] < self.wma):
                    self.log(f"SIGNAL NUMBER #{self.signal_number}")
                    self.signal_number += 1
                    # if (self.dataclose[0] < self.dick.lines.bbb) and (self.dataclose[0] < self.wma):
                    self.sizer()
                    self.log(
                        f"(1) SIGNAL NOTICE: Sell {self.size} shares of {self.params.symbol} at {self.data.close[0]:.2f}"
                    )
                    # self.sell(size=self.size)
                    # self.currency_format = str(self.size)[::-1].find('.')
                    self.sell(exectype=bt.Order.Market, size=self.size)
                    if self.params.trstop == 1:
                        self.buy(
                            exectype=bt.Order.StopTrail,
                            size=self.size,
                            trailpercent=self.params.trstop_percent,
                        )
                    self.wallet = self.broker.getvalue()

        # CLOSE POSITIONS
        else:
            # CLOSE LONG
            if self.position.size > 0:
                # TAKE PROFIT
                if self.datahigh[0] >= self.executed_price * (
                    1 + self.params.takeprofit
                ):
                    # self.close()
                    self.sell(
                        exectype=bt.Order.Limit,
                        size=self.size,
                        price=self.executed_price * (1 + self.params.takeprofit),
                    )
                    self.log(
                        f"(3) CLOSE LONG position at {self.executed_price * (1 + self.params.takeprofit):.2f}"
                    )
                    self.profit_loss = "profit"
                elif (
                    (self.params.emergency_exit == 1)
                    and (self.rsi < self.params.rsi_value_long)
                    and (self.datahigh[0] > self.executed_price)
                ):
                    self.close()
                    self.log(
                        f"(3) EMERGENCY EXIT: CLOSE LONG position at {self.executed_price:.2f}"
                    )
                    self.profit_loss = "profit"
                # STOP LOSS
                if self.datalow[0] <= self.executed_price * (1 - self.params.stoploss):
                    self.close()
                    # self.sell(exectype=bt.Order.Market, size=self.size)
                    self.log(
                        f"(3) CLOSE LONG position at {self.executed_price * (1 - self.params.stoploss):.2f}"
                    )
                    self.profit_loss = "loss"

            # CLOSE SHORT
            if self.params.short_positions:
                if self.position.size < 0:
                    # TAKE PROFIT
                    if self.datalow[0] <= self.executed_price * (
                        1 - self.params.takeprofit
                    ):
                        # self.close()
                        self.buy(
                            exectype=bt.Order.Limit,
                            size=self.size,
                            price=self.executed_price * (1 - self.params.takeprofit),
                        )
                        self.log(
                            f"(3) CLOSE SHORT position at {self.executed_price * (1 - self.params.takeprofit):.2f}"
                        )
                        self.profit_loss = "profit"
                    elif (
                        (self.params.emergency_exit == 1)
                        and (self.rsi > self.params.rsi_value_short)
                        and (self.datahigh[0] < self.executed_price)
                    ):
                        self.close()
                        self.log(
                            f"(3) EMERGENCY EXIT: CLOSE SHORT position at {self.executed_price:.2f}"
                        )
                        self.profit_loss = "profit"
                    # STOP LOSS
                    if self.datahigh[0] >= self.executed_price * (
                        1 + self.params.stoploss
                    ):
                        self.close()
                        # self.buy(exectype=bt.Order.Market, size=self.size)
                        self.log(
                            f"(3) CLOSE SHORT position at {self.executed_price * (1 + self.params.stoploss):.2f}"
                        )
                        self.profit_loss = "loss"

    def stop(self):
        self.log(f'\n{50*"+"}\n')
        self.log(
            f"STOP RESULTS : \n\n* factor : {self.p.factor}\n* multiplier : {self.p.multiplier} \n* period : {self.p.period}",
            doprint=False,
        )
        self.log(f'\n{50*"+"}\n')
