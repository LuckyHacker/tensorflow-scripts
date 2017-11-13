
class StockTradingSimulation:

    def __init__(self, diff, ohlc, dataset, starting_capital=5000,
                    trading_fee=0.2, min_fee=9, req_diff=0.01, bad_luck=0.05,
                    price="close"):

        self.diff = diff
        self.ohlc = ohlc[-len(self.diff):]
        self.starting_capital = starting_capital
        self.trading_fee = trading_fee / 100
        self.min_fee = min_fee
        self.req_diff = req_diff
        self.bad_luck = bad_luck
        self.dataset = dataset
        self.price = price
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

        for i in range(len(self.diff) - 1):
            # + 1 day (next day prices will be used)
            self.current_open_price = self.ohlc[i + 1][0]
            self.current_high_price = self.ohlc[i + 1][1]
            self.current_low_price = self.ohlc[i + 1][2]
            self.current_close_price = self.ohlc[i + 1][3]

            self.current_low_avg_price = self.current_low_price + (self.current_high_price - self.current_low_price) * (0.50 - self.bad_luck)
            self.current_high_avg_price = self.current_low_price + (self.current_high_price - self.current_low_price) * (0.50 + self.bad_luck)
            self.fee_amount = 0
            self.used_prices = {"open": self.current_open_price,
                                "close": self.current_close_price,
                                "low_avg": self.current_low_avg_price,
                                "high_avg": self.current_high_avg_price,
                                }

            print("Day {}".format(i + 1))
            if self.diff[i] < -self.req_diff and self.num_shares > 0:
                self.state = "Sell"
                sell_list.append(i)
                self.current_price = self.used_prices[self.price]
                self.fee_amount = self.num_shares * self.current_price * self.trading_fee

                if self.fee_amount < self.min_fee:
                    self.fee_amount = self.min_fee

                self.current_money = self.num_shares * self.current_price - self.fee_amount
                self.total_paid_fee += self.fee_amount
                self.num_shares = 0

            elif self.diff[i] > self.req_diff and self.num_shares == 0:
                self.state = "Buy"
                buy_list.append(i)
                self.current_price = self.used_prices[self.price]
                self.fee_amount = self.current_money * self.trading_fee

                if self.fee_amount < self.min_fee:
                    self.fee_amount = self.min_fee

                self.num_shares = (self.current_money - self.fee_amount) / self.current_price
                self.total_paid_fee += self.fee_amount
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
            print("Total paid fee: {}".format(round(self.total_paid_fee)))
            print("Total worth: {}".format(round(self.total_worth)))
            print("Total profit: {} %".format(round(self.total_profit)))
            print("")
            print("")

        total_worth_wo = (self.starting_capital / self.ohlc[0][0]) * self.ohlc[-1][0]
        total_profit_wo = total_worth_wo / self.starting_capital * 100 - 100
        print("Total profit without trading: {}%".format(round(total_profit_wo)))

        return sell_list, buy_list, self.total_profit

    def _norm_to_original_diff(self, scalar):
        return scalar * (self.dataset["Close"].max() - self.dataset["Close"].min())

    def _norm_to_original(self, scalar):
        return scalar * (self.dataset["Close"].max() - self.dataset["Close"].min()) + self.dataset["Close"].min()
