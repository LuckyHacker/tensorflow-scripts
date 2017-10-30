import tensorflow as tf
import pandas as pd
import numpy as np
import time


num_features = 30
truncated_backprop_length = 1
state_size = 75
num_classes = 1
batch_size = 10
train_size = 0.9
num_epochs = 100
learning_rate = 0.001

csv_file_path = "data/creditcard_sampled.csv"

dataset = pd.read_csv(csv_file_path)
train_length = int(len(dataset.index) * train_size)
test_length = int(len(dataset.index) * (1.0 - train_size))
num_batches = train_length // batch_size // truncated_backprop_length

features = ["Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
            ]

def prepare_data():
    #datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetNorm = dataset.sample(frac=1)
    datasetTrain = datasetNorm[dataset.index < train_length]
    datasetTest = datasetNorm[dataset.index >= train_length]

    xTrain = datasetTrain[features].as_matrix()
    yTrain = datasetTrain[["Class"]].as_matrix()
    xTest = datasetTest[features].as_matrix()
    yTest = datasetTest[["Class"]].as_matrix()

    return xTrain, yTrain, xTest, yTest


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


def train(  loss, train_step, prediction, last_label, last_state,
            batchX_placeholder, batchY_placeholder, xTrain, yTrain,
            xTest, yTest):
    loss_list = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Train data length: %d' % train_length)
        print('Test data length: %d' % (test_length+1))

        # Train
        for epoch_idx in range(num_epochs):
            begin_time = time.time()
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

            end_time = time.time() - begin_time
            print('Epoch: %d, loss: %.6f, time: %.2fs' % (epoch_idx + 1, _loss, end_time))


        pred_list = []
        for test_idx in range(len(xTest) - truncated_backprop_length + 1):
            testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1, truncated_backprop_length, num_features))
            testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1, truncated_backprop_length, num_classes))

            feed = {batchX_placeholder : testBatchX,
                    batchY_placeholder : testBatchY}

            _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
            pred_list.append(test_pred[-1][0])


        pred_list = list(map(lambda x: int(round(x)), pred_list))
        correct_preds = 0

        for i in range(len(yTest)):
            if yTest[i][0] == pred_list[i]:
                correct_preds += 1


        accuracy = correct_preds / len(yTest) * 100
        print("Accuracy: {}%".format(accuracy))
        return pred_list


if __name__ == "__main__":
    loss, train_step, prediction, last_label, last_state, batchX_placeholder, batchY_placeholder = model()
    xTrain, yTrain, xTest, yTest = prepare_data()
    pred_list = train(  loss, train_step, prediction, last_label,
                        last_state, batchX_placeholder, batchY_placeholder,
                        xTrain, yTrain, xTest, yTest)
