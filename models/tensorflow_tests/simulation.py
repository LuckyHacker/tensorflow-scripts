
class StockTradingSimulation:

    def __init__(self, diff, ohlc, dataset, starting_capital=5000,
                    trading_fee=0.2, min_fee=9, req_diff=0.01):

        self.diff = diff
        self.ohlc = ohlc[-(len(self.diff)+1):]
        self.starting_capital = starting_capital
        self.trading_fee = trading_fee / 100
        self.min_fee = min_fee
        self.req_diff = req_diff
        self.dataset = dataset
        self.state = "Idle"

        self.num_shares = 0
        self.total_paid_fee = 0
        self.best_profit = 0
        self.worst_profit = 0
        self.fee_amount = 0
        self.current_price = 0
        self.trading_fee_ratio = 1 - self.trading_fee
        self.current_money = self.starting_capital

    def run(self):
        sell_list = []
        buy_list = []

        for i in range(len(self.diff)):
            self.current_high_price = self.ohlc[i + 1][1]
            self.current_low_price = self.ohlc[i + 1][2]
            self.current_open_price = self.ohlc[i + 1][0]
            self.current_low_avg_price = self.current_low_price + (self.current_high_price - self.current_low_price) * 0.45
            self.current_high_avg_price = self.current_low_price + (self.current_high_price - self.current_low_price) * 0.55
            self.fee_amount = 0

            print("Day {}".format(i + 1))
            if self.diff[i] < -self.req_diff and self.num_shares > 0:
                self.state = "Sell"
                sell_list.append(i)
                self.current_price = self.current_open_price
                self.fee_amount = self.num_shares * self.current_price * self.trading_fee

                if self.fee_amount < self.min_fee:
                    self.fee_amount = self.min_fee

                self.current_money = self.num_shares * self.current_price - self.fee_amount
                self.total_paid_fee += self.fee_amount
                self.num_shares = 0

            elif self.diff[i] > self.req_diff and self.num_shares == 0:
                self.state = "Buy"
                buy_list.append(i)
                self.current_price = self.current_open_price
                self.fee_amount = self.current_money * self.trading_fee

                if self.fee_amount < self.min_fee:
                    self.fee_amount = self.min_fee

                self.total_paid_fee += self.fee_amount
                self.num_shares = (self.current_money - self.fee_amount) / self.current_price
                self.current_money = 0

            else:
                self.state = "Idle"


            self.total_worth = self.current_money + (self.num_shares * self.current_low_price)
            self.total_profit = self.total_worth / self.starting_capital * 100 - 100

            if self.total_profit > self.best_profit:
                self.best_profit = self.total_profit

            if self.total_profit < self.worst_profit:
                self.worst_profit = self.total_profit

            print("State: {}".format(self.state))
            print("Predicted diff (Normalized): {0:.3f}".format(self.diff[i]))
            print("Predicted diff (Original): {0:.2f}".format(self._norm_to_original_diff(self.diff[i])))
            print("")
            print("Current price: {}".format(round(self.current_price)))
            print("Current money: {}".format(round(self.current_money)))
            print("Current shares: {}".format(round(self.num_shares)))
            print("Trading fee: {}".format(round(self.fee_amount)))
            print("Total worth: {}".format(round(self.total_worth)))
            print("Total profit: {} %".format(round(self.total_profit)))
            print("")
            print("")

        return sell_list, buy_list, self.total_profit

    def _norm_to_original_diff(self, scalar):
        return scalar * (self.dataset["Close"].max() - self.dataset["Close"].min())

    def _norm_to_original(self, scalar):
        return scalar * (self.dataset["Close"].max() - self.dataset["Close"].min()) + self.dataset["Close"].min()
