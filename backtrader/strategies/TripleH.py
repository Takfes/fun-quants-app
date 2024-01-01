import numpy as np

import backtrader as bt

debug = False


class PAT(bt.Indicator):
    lines = ("ph", "pl")

    plotinfo = dict(plot=True, subplot=False, plotlinelabels=True)
    plotlines = dict(
        ph=dict(
            marker="o", markersize=6.0, color="blue", fillstyle="full", ls=""
        ),  # $\u21E9$
        pl=dict(
            marker="o", markersize=6.0, color="red", fillstyle="full", ls=""
        ),  # $\u21E7$
    )

    params = (
        ("atr_period", 170),
        ("pivot_period", 3),
        ("factor", 6.5),
        ("printlog", True),
        ("barplot", False),  # plot above/below max/min for clarity in bar plot
        ("bardist", 0.005),  # distance to max/min in absolute perc
    )

    def __init__(self):
        # indicate min period ; i.e. buffering period
        self.buffering_period = (self.p.pivot_period * 2) + 1
        self.addminperiod(self.buffering_period)
        self.ATR = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.center = 0
        self.lastpp = 0
        self.tup = 0
        self.tdn = 0
        self.trend = [0, 1]
        self.ph = self.pl = 0

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def next(self):
        # grab necessary info
        highp = np.array(self.data.high.get(size=self.buffering_period))
        lowp = np.array(self.data.low.get(size=self.buffering_period))
        closep = np.array(self.data.close.get(size=self.buffering_period))

        # get highest high or lowest low position
        highest_bar_position = highp.argmax()
        lowest_bar_position = lowp.argmin()

        # check for pivot high
        if highest_bar_position == (self.p.pivot_period + 1):
            self.lines.ph[-(self.p.pivot_period + 1)] = self.ph = np.max(highp)
            self.lastpp = np.max(highp)
            # self.cacheh = len(self.datas[0])

        # check for pivot low
        if lowest_bar_position == (self.p.pivot_period + 1):
            self.lines.pl[-(self.p.pivot_period + 1)] = self.pl = np.min(lowp)
            self.lastpp = np.min(lowp)
            # self.cachel = len(self.datas[0])

        if self.lastpp != 0:
            # calculate center
            if self.center == 0:
                self.center = self.lastpp
            else:
                self.center = (self.center * 2 + self.lastpp) / 3

            # calculate up and down
            self.up = self.center - (self.p.factor * self.ATR[0])
            self.dn = self.center + (self.p.factor * self.ATR[0])

            # calculate trend up
            if closep[-1] > self.tup:
                self.tup = max(self.tup, self.up)
            else:
                self.tup = self.up

            # calculate trend down
            if closep[-1] < self.tdn:
                self.tup = min(self.tdn, self.dn)
            else:
                self.tdn = self.dn

            # calculate trend
            if closep[0] > self.tdn:
                self.trend.append(1)
            elif closep[0] < self.tup:
                self.trend.append(-1)
            else:
                self.trend.append(1)

            # calculate trigger line
            if self.trend[-1] == 1:
                self.tl = self.tup
            else:
                self.tl = self.tdn

        if debug:
            self.log(f"length : {len(self.datas[0])}")
            self.log(f"highp : {highp[0]}")
            self.log(highp)
            self.log(f"highest_bar_position : {highest_bar_position}")
            self.log(f"> ph : {self.ph}")
            self.log(f"> pl : {self.pl}")
            self.log(50 * "=")


class TripleH(bt.Strategy):
    params = (
        ("symbol", "unknown"),
        ("cash", 1000),
        ("risk", 0.25),
        ("stoploss", 0.01),
        ("takeprofit", 0.01),
        ("trstop", 0),
        ("trstop_percent", 0.005),
        ("short_positions", 0),
        ("atr_period", 170),
        ("pivot_period", 3),
        ("factor", 6.5),
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

        # indicators
        self.ATR = bt.indicators.ATR(self.datas[1], period=self.p.atr_period)
        pat = self.pat = PAT(
            self.datas[1],
            atr_period=self.p.atr_period,
            pivot_period=self.p.pivot_period,
            factor=self.p.factor,
        )
        pat.plotinfo.subplot = False

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
            if (self.pat.trend[-1] == 1) & (self.pat.trend[-2] == -1):
                self.log(f"SIGNAL NUMBER #{self.signal_number}")
                self.signal_number += 1
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
                if (self.pat.trend[-1] == -1) & (self.pat.trend[-2] == 1):
                    self.log(f"SIGNAL NUMBER #{self.signal_number}")
                    self.signal_number += 1
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
                ##################### TO USE DIFFERENT RESAMPLE PERIOD #####################
                #
                # self.do open
                # self.dh high
                # self.dl low
                # self.dc close
                #
                # example : self.dc[0] current data close for resampled data
                #
                ############################################################################

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
        self.log(f'\n{50 * "+"}\n')
        self.log(
            f"STOP RESULTS : \n\n* atr_period : {self.p.atr_period}\n* pivot_period : {self.p.pivot_period} \n* factor : {self.p.factor}",
            doprint=False,
        )
        self.log(f'\n{50 * "+"}\n')
