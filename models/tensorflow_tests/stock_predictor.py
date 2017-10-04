import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

outfile = "prediction.png"
outfolder = "output"
infile = "TELIA1.HE_alltargets.csv"
infolder = "stock" # ("stock" / "currency")

learning_rate = 0.001
num_epochs = 10
batch_size = 1
train_size = 0.9
truncated_backprop_length = 1
state_size = 12
num_features = 4
num_classes = 4


dataset = pd.read_csv('../data/{}/{}'.format(infolder, infile))
train_length = int(len(dataset.index) * train_size)
test_length = int(len(dataset.index) * (1.0 - train_size))
num_batches = train_length // batch_size // truncated_backprop_length


def prepare_data():
    datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetTrain = datasetNorm[dataset.index < train_length]
    datasetTest = datasetNorm[dataset.index >= train_length]


    try:
        xTrain = datasetTrain[['Open','High','Low','Close']].as_matrix()
        yTrain = datasetTrain[['OpenTarget','HighTarget','LowTarget','CloseTarget']].as_matrix()
        xTest = datasetTest[['Open','High','Low','Close']].as_matrix()
        yTest = datasetTest[['OpenTarget','HighTarget','LowTarget','CloseTarget']].as_matrix()
        test_prices = datasetTest['Close'].as_matrix()
        print("Using OHLC data")
    except:
        xTrain = datasetTrain[['Close','MACD','Stochastics','ATR']].as_matrix()
        yTrain = datasetTrain[['CloseTarget', 'MACDTarget', 'StochasticsTarget', 'ATRTarget']].as_matrix()
        xTest = datasetTest[['Close','MACD','Stochastics','ATR']].as_matrix()
        yTest = datasetTest[['CloseTarget', 'MACDTarget', 'StochasticsTarget', 'ATRTarget']].as_matrix()
        test_prices = datasetTest['Close'].as_matrix()
        print("Using technical indicators data")

    return xTrain, xTest, yTrain, yTest, test_prices


def model():
    batchX_placeholder = tf.placeholder(dtype=tf.float32,
                                        shape=[None, truncated_backprop_length, num_features],
                                        name='data_ph')
    batchY_placeholder = tf.placeholder(dtype=tf.float32,
                                        shape=[None, truncated_backprop_length, num_classes],
                                        name='target_ph')
    W2 = tf.Variable(   initial_value=np.random.rand(state_size, num_classes),
                        dtype=tf.float32)
    b2 = tf.Variable(   initial_value=np.random.rand(1, num_classes),
                        dtype=tf.float32)
    labels_series = tf.unstack(batchY_placeholder, axis=1)
    cell = tf.contrib.rnn.LSTMCell(num_units=state_size)
    states_series, current_state = tf.nn.dynamic_rnn(   cell=cell,
                                                        inputs=batchX_placeholder,
                                                        dtype=tf.float32)
    states_series = tf.transpose(states_series, [1,0,2])
    last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0]-1)
    last_label = tf.gather(params=labels_series, indices=len(labels_series)-1)
    weight = tf.Variable(tf.truncated_normal([state_size, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    prediction = tf.matmul(last_state, weight) + bias
    loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return (loss, train_step, prediction, last_label,
            last_state, batchX_placeholder, batchY_placeholder)


def train_and_test( loss, train_step, prediction, last_label, last_state,
                    batchX_placeholder, batchY_placeholder,
                    xTrain, xTest, yTrain, yTest):
    loss_list = []
    test_days_pred_list = []
    test_day_pred_list = []
    test_days_differences = []
    test_day_differences = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Train data length: %d' % train_length)
        print('Test data length: %d' % (test_length+1))
        _loss = 0

        # Train
        for epoch_idx in range(num_epochs):
            print('Epoch %d, loss %.6f' % (epoch_idx + 1, _loss))
            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length * batch_size

                batchX = xTrain[start_idx:end_idx,:].reshape(batch_size,truncated_backprop_length,num_features)
                batchY = yTrain[start_idx:end_idx].reshape(batch_size,truncated_backprop_length,num_classes)

                _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                    fetches = [loss, train_step, prediction, last_label, prediction],
                    feed_dict = {   batchX_placeholder : batchX,
                                    batchY_placeholder : batchY}
                )

                loss_list.append(_loss)

        # Test len(xTest) days
        for test_idx in range(len(xTest) - truncated_backprop_length):
            if test_idx == 0:
                testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1, truncated_backprop_length, num_classes))
                testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1, truncated_backprop_length, num_features))
                testBatchX = testBatchX.tolist()
            else:
                testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1, truncated_backprop_length, num_classes))
                test_pred = test_pred[0].tolist()
                testBatchX[0].append(test_pred)
                testBatchX[0].pop(0)

            test_pred = prediction.eval(feed_dict={batchX_placeholder : testBatchX})
            test_days_pred_list.append(test_pred[-1][0])

            if test_idx > 0:
                test_days_differences.append(test_days_pred_list[-1] - yTest[test_idx - 1][0])

        # Test per day
        for test_idx in range(len(xTest)):
            testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1, truncated_backprop_length, num_features))
            testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1, truncated_backprop_length, num_classes))

            feed = {batchX_placeholder : testBatchX,
                    batchY_placeholder : testBatchY}

            _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
            test_day_pred_list.append(test_pred[-1][0])

            if test_idx > 0:
                test_day_differences.append(test_day_pred_list[-1] - test_day_pred_list[-2])

    return test_day_pred_list, test_days_pred_list, test_day_differences, test_days_differences


def plot_prediction(day, days, test_prices, day_differences, days_differences,
                    buy_list, sell_list, profits, starting_funds, trading_fee, min_fee, req_diff):

    fig = plt.figure(num=None, figsize=(22, 12), dpi=80, facecolor="white")
    fig.canvas.set_window_title("Stock prediction {}".format(infile))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    day_price_error = sum(day_differences) / len(day_differences) * (dataset["Close"].max() - dataset["Close"].min())
    days_price_error = sum(days_differences) / len(days_differences) * (dataset["Close"].max() - dataset["Close"].min())

    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price")
    ax1.set_title("Close price prediction per day")
    ax1.text(0.5, 0.95, "Average price error: %.6f" % day_price_error,
                    fontsize=16,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax1.transAxes)
    ax1.plot(test_prices, label='Price', color='blue')
    ax1.plot(day, label='Predicted', color='red')
    ax1.legend(loc='upper left')

    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price")
    ax2.set_title("Prediction for {} days".format(len(test_prices)))
    ax2.text(0.5, 0.95, "Average price error: %.6f" % (days_price_error),
                    fontsize=16,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
    ax2.plot(test_prices, label='Price', color='blue')
    ax2.plot(days, label='Predicted', color='red')
    ax2.legend(loc='upper left')

    ax3.set_xlabel("Days")
    ax3.set_ylabel("Price")
    ax3.set_title("Virtual money simulation")
    ax3.plot(test_prices, label='Price (buy/sell)', color='blue', markevery=sell_list, markerfacecolor="red", markeredgecolor="red", marker="o")
    ax3.plot(test_prices, color='blue', markevery=buy_list, markerfacecolor="green", markeredgecolor="green", marker="o")
    ax3.legend(loc='upper left')

    ax4.set_xlabel("Profit percent")
    ax4.set_ylabel("Starting money")
    ax4.set_title("Starting price compared to profit")
    ax4.text(0.5, 0.85, "Trading fee: {} %\nMin fee: {}\nReq diff: {}".format(trading_fee * 100, min_fee, req_diff),
                    fontsize=16,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax4.transAxes)
    ax4.plot(profits, starting_funds, label='Profits', color='blue')
    ax4.legend(loc='upper left')

    plt.figtext(0.005, 0.8, """Learning Rate = {}\nEpochs = {}\nBatch Size = {}\nTrain Size = {}\nTruncated Backprop = {}\nState Size = {}""".format(
                        learning_rate, num_epochs, batch_size, train_size,
                        truncated_backprop_length, state_size), bbox={'facecolor': 'darkblue', 'alpha': 0.5, 'pad': 10},
                        fontsize=11)

    plt.savefig(os.path.join(outfolder, outfile))
    plt.show()

def norm_to_original(scalar):
    return scalar * (dataset["Close"].max() - dataset["Close"].min()) + dataset["Close"].min()

def simulate_profit(day_diff, test_prices, starting_money=200, trading_fee=0.0006, min_fee=3, req_diff = 0.01):
    num_shares = 0
    total_paid_fee = 0
    best_profit = 0
    worst_profit = 0
    fee_amount = 0
    trading_fee_ratio = 1 - trading_fee
    current_money = starting_money

    buy_list = []
    sell_list = []
    for count, d in enumerate(day_diff):
        current_price = norm_to_original(test_prices[count])
        print("Day {}".format(count + 1))
        print("Diff: {}".format(d))
        if d < -req_diff and num_shares > 0:
            print("State: Selling")
            sell_list.append(count)
            fee_amount = num_shares * current_price * trading_fee
            if fee_amount < min_fee:
                fee_amount = min_fee
            current_money = num_shares * current_price - fee_amount
            total_paid_fee += fee_amount
            num_shares = 0
        elif d > req_diff and num_shares == 0:
            print("State: Buying")
            buy_list.append(count)
            fee_amount = current_money * trading_fee
            if fee_amount < min_fee:
                fee_amount = min_fee
            total_paid_fee += fee_amount
            num_shares = (current_money - fee_amount) / current_price
            current_money = 0
        else:
            print("State: Idle")

        total_worth = current_money + (num_shares * current_price)
        total_profit = total_worth / starting_money * 100 - 100
        if total_profit > best_profit:
            best_profit = total_profit

        if total_profit < worst_profit:
            worst_profit = total_profit

        print("Current price: {}\nCurrent money: {}\nCurrent shares: {}\nTrading fee: {}\nTotal worth: {}\nTotal profit: {} %".format(
            round(current_price),
            round(current_money),
            round(num_shares),
            round(fee_amount),
            round(total_worth),
            round(total_profit)))
        print("")

    print("Total paid fee: {}\nBest profit: {} %\nWorst profit: {} %\nTotal money earned: {}".format(
        total_paid_fee,
        best_profit,
        worst_profit,
        total_worth - starting_money))

    return buy_list, sell_list, total_profit

if __name__ == "__main__":
    outfile = ".".join(infile.split(".")[:-1]) + "_" + outfile
    trainX, testX, trainY, testY, prices = prepare_data()
    loss, train_step, prediction, last_label, last_state, placeholder_x, placeholder_y = model()
    day, days, test_day_differences, test_days_differences = train_and_test( loss, train_step, prediction, last_label, last_state,
                                placeholder_x, placeholder_y,
                                trainX, testX, trainY, testY)

    trading_fee = 0.0006
    min_fee = 3
    req_diff = 0.005
    starting_funds = list(range(50, 5050, 50))
    profits = []
    for starting_money in starting_funds:
        buy_list, sell_list, total_profit = simulate_profit(test_day_differences,
                                                            prices, starting_money,
                                                            trading_fee=trading_fee,
                                                            min_fee=min_fee,
                                                            req_diff=req_diff)

        profits.append(total_profit)

    plot_prediction(day, days, prices, test_day_differences,
                    test_days_differences, buy_list, sell_list,
                    profits, starting_funds, trading_fee, min_fee, req_diff)