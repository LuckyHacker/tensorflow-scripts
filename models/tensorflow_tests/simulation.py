
class StockTradingSimulation:

    def __init__(self, diff, close_prices, open_prices, high_prices,
                    low_prices, starting_capital=5000, trading_fee=0.2,
                    min_fee=9, req_diff=0.01):

        self.diff = diff
        self.close_prices = close_prices
        self.open_prices = open_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.starting_capital = 5000
        self.trading_fee = trading_fee / 100

    def run(self):
        for i in range(len(close_prices)):
            pass

    def norm_to_original_diff(self, scalar):
        return scalar * (dataset["Close"].max() - dataset["Close"].min())

    def norm_to_original(self, scalar):
        return scalar * (dataset["Close"].max() - dataset["Close"].min()) + dataset["Close"].min()
