import backtrader as bt
import pandas as pd
from strategies.MACDCross import MACDCross

# plt.ion()

df = pd.read_pickle("data/eth_1h_ohlcv.pkl").drop("symbol", axis=1)

data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()

cerebro.adddata(data)
cerebro.addstrategy(MACDCross)

starting_cash = 1000
# commision = 0.0025

cerebro.broker.set_cash(starting_cash)
# cerebro.broker.setcommission(commission=commision)

print(f"Starting Cash: {cerebro.broker.startingcash}")

cerebro.run()

profit = cerebro.broker.getvalue() - cerebro.broker.startingcash
roi = profit / cerebro.broker.startingcash

print("=" * 50)
print(f"Final Cash: {cerebro.broker.cash}")
print(f"Final Value: {cerebro.broker.getvalue()}")
print(f"Profit: {profit:.2f}")
print(f"ROI: {roi:.2%}")

cerebro.plot()

# fig = cerebro.plot()[0][0]
# fig.savefig("backtrader_plot.png", dpi=300)
# plt.show()
