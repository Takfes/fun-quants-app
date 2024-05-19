import backtrader as bt


class MACDCross(bt.Strategy):
    params = (
        ("fastperiod", 12),
        ("slowperiod", 26),
        ("signalperiod", 9),
        ("macd_cross_macdsignal", 12),
        ("macd_cross_zero", 3),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED : {order.executed.price}")
            elif order.issell():
                self.log(f"SELL EXECUTED : {order.executed.price}")
            self.bar_executed = len(self)
        self.order = None

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.macd = None
        self.macdsignal = None
        self.macdhist = None
        self.macd, self.macdsignal, self.macdhist = bt.talib.MACD(
            self.dataclose,
            fastperiod=26,
            slowperiod=12,
            signalperiod=9,
            # fastperiod=self.params.fastperiod,
            # slowperiod=self.params.slowperiod,
            # signalperiod=self.params.signalperiod,
        )
        # self.macd_cross_macdsignal = bt.ind.CrossOver(self.macd, self.macdsignal)
        # self.macd_cross_zero = bt.ind.CrossOver(self.macd, 0)

    def next(self):
        self.log("Close, %.4f" % self.datas[0].close[0])

        if not self.position:
            if self.macd[0] > self.dataclose[0]:
                self.log(f"BUY CREATE for {self.params.ticker} @ {self.dataclose[0]}")
                self.order = self.buy()

        else:
            if len(self) >= (self.bar_executed + 5):
                self.log(f"SELL CREATE for {self.params.ticker} @ {self.dataclose[0]}")
                self.order = self.close()
