import math

import backtrader as bt


class GoldenCross(bt.Strategy):
    params = (("symbol", "None"), ("risk", 0.1), ("fast", 100), ("slow", 300))

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.fast,
            plotname=f"{self.p.fast} moving average",
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.slow,
            plotname=f"{self.p.slow} moving average",
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    # def print_parameters(self):
    #     print(f'\n====================================\n')
    #     print(f'This is the GoldenCross print_parameters :')
    #     for k,v in self.params._getitems():
    #         print(f'* {k} : {v}')
    #     print(f'\n====================================\n')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")  # Print date and close

    def sizer(self):
        amount_to_invest = self.params.risk * self.broker.cash
        size = self.size = round((amount_to_invest / self.datas[0].close), 3)
        return size

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                # self.log(f'BUY EXECUTED : {order.executed.price}')
                self.executed_price = order.executed.price
            elif order.issell():
                # self.log(f'SELL EXECUTED : {order.executed.price}')
                self.executed_price = order.executed.price
            self.bar_executed = len(self)
        self.order = None

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy(size=self.sizer())

        if self.crossover < 0:
            self.close()
