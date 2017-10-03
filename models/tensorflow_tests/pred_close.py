import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import datetime as dt
from pandas_datareader import data, wb

pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


learning_rate = 0.001
num_epochs = 10
batch_size = 1
truncated_backprop_length = 1
state_size = 12
num_features = 4
num_classes = 4


def get_data():
    start_date = dt.datetime(1980, 1, 1)
    dat = data.DataReader(sys.argv[1], 'yahoo', start_date, dt.datetime.today())
    dat.to_csv('data/{}.csv'.format(stock), mode='w', header=True)

def prepare_stock():
    stock_csv = "data/{}.csv".format(stock)
    outfile = "data/{}_alltargets.csv".format(stock)
    datefile = outfile.split(".")[0] + "_latest_date.txt"

    def MACD(df, period1, period2, periodSignal):
        EMA1 = pd.DataFrame.ewm(df, span=period1).mean()
        EMA2 = pd.DataFrame.ewm(df, span=period2).mean()
        MACD = EMA1 - EMA2
        Signal = pd.DataFrame.ewm(MACD,periodSignal).mean()
        Histogram = MACD - Signal
        return Histogram

    def stochastics_oscillator(df, period):
        l, h = pd.DataFrame.rolling(df, period).min(), pd.DataFrame.rolling(df, period).max()
        k = 100 * (df - l) / (h - l)
        return k

    def ATR(df, period):
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        TR = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        return TR.to_frame()

    def adjust_csv():
        with open(outfile, "r") as f:
            lines = f.readlines()[1:]

        lines.insert(0, "Close,CloseTarget,MACD,MACDTarget,Stochastics,StochasticsTarget,ATR,ATRTarget\n")

        # Remove entries where stochastics could not be calculated
        new_lines = []
        for line in lines:
            for count, col in enumerate(line.split(",")):
                if count == 4:
                    if col == "":
                        break
                    else:
                        new_lines.append(line)



        with open(outfile, "w") as f:
            for line in new_lines:
                f.write(line)

    df = pd.read_csv(stock_csv, usecols=[0,1,2,3,4])

    latest_date = df['Date'][df.index[-1]]
    dfPrices = df['Close']
    dfPriceShift = dfPrices.shift(-1)
    macd = MACD(dfPrices, 12, 26, 9)
    macdShift = macd.shift(-1)
    stochastics = stochastics_oscillator(dfPrices, 14)
    stochasticsShift = stochastics.shift(-1)
    atr = ATR(df, 14)
    atrShift = atr.shift(-1)

    final_data = pd.concat([dfPrices,
                            dfPriceShift,
                            macd,
                            macdShift,
                            stochastics,
                            stochasticsShift,
                            atr,
                            atrShift
                            ], axis=1)

    final_data.to_csv(outfile, index=False)
    with open(datefile, "w") as f:
        f.write(latest_date)

    adjust_csv()

def prepare_data():
    datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetTrain = datasetNorm
    datasetTest = datasetNorm.iloc[[-2, -1]]

    xTrain = datasetTrain[['Close','MACD','Stochastics','ATR']].as_matrix()
    yTrain = datasetTrain[['CloseTarget', 'MACDTarget', 'StochasticsTarget', 'ATRTarget']].as_matrix()
    xTest = datasetTest[['Close','MACD','Stochastics','ATR']].as_matrix()
    yTest = datasetTest[['CloseTarget', 'MACDTarget', 'StochasticsTarget', 'ATRTarget']].as_matrix()
    test_prices = datasetTest['Close'].as_matrix()

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

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Train data length: %d' % train_length)
        print('Test data length: %d' % test_length)
        _loss = 0

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

                if str(_loss) == "nan":
                    print("WARNING: Incomplete data! Quitting.")
                    sys.exit()

                loss_list.append(_loss)


        preds = []
        for i in range(len(xTest)):
            testBatchX = xTest[i].reshape((1, truncated_backprop_length, num_features))
            preds.append(prediction.eval(feed_dict={batchX_placeholder : testBatchX})[0][0])

        diff = preds[-1] - preds[-2]
        estimate_action(diff)

def norm_to_original(scalar):
    return scalar * (dataset["Close"].max() - dataset["Close"].min()) + dataset["Close"].min()

def estimate_action(d):
    req_diff = 0.01

    print("")
    print("Prediction for next close:")
    print("Latest_date: {}".format(latest_date))
    print("Change: {}".format(norm_to_original(d)))

    if d < -req_diff:
        print("Action: Sell")
    elif d > req_diff:
        print("Action: Buy")
    else:
        print("Action: Idle")

def help_p():
    print("Usage:")
    print("python3 {} <stock name>".format(sys.argv[0]))

if __name__ == "__main__":
    try:
        stock = sys.argv[1]
    except:
        help_p()
        sys.exit()

    infile = "data/{}_alltargets.csv".format(stock)
    datefile = infile.split(".")[0] + "_latest_date.txt"

    get_data()
    prepare_stock()

    with open(datefile, "r") as f:
        latest_date = f.read()

    dataset = pd.read_csv(infile)
    train_length = len(dataset.index) - 2
    test_length = 2
    num_batches = train_length // batch_size // truncated_backprop_length

    trainX, testX, trainY, testY, prices = prepare_data()
    loss, train_step, prediction, last_label, last_state, placeholder_x, placeholder_y = model()
    train_and_test( loss, train_step, prediction, last_label, last_state,
                                placeholder_x, placeholder_y,
                                trainX, testX, trainY, testY)
