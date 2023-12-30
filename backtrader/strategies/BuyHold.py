import backtrader as bt


class BuyHold(bt.Strategy):
    def next(self):
        if self.position.size == 0:
            size = int(self.broker.get_cash() / self.data)
            self.buy(size=size)
