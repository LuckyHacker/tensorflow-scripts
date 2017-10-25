import tensorflow as tf
import pandas as pd
import numpy as np
import time


num_features = 27
truncated_backprop_length = 1
state_size = 75
num_classes = 1
batch_size = 10
train_size = 0.9
num_epochs = 100
learning_rate = 0.001

csv_file_path = "data/access.csv"
log_file_path = "data/access.log"

dataset = pd.read_csv(csv_file_path)
train_length = int(len(dataset.index) * train_size)
test_length = int(len(dataset.index) * (1.0 - train_size))
num_batches = train_length // batch_size // truncated_backprop_length

features = ["date",
            "time",
            "request_header_user_agent__browser__version_string",
            "request_header_referer",
            "request_url",
            "request_url_username",
            "remote_host",
            "request_header_user_agent__os__version_string",
            "request_url_password",
            "request_url_path",
            "request_url_port",
            "request_url_query",
            "request_http_ver",
            "response_bytes_clf",
            "request_url_scheme",
            "request_header_user_agent__is_mobile",
            "request_url_netloc",
            "request_header_user_agent__browser__family",
            "request_url_hostname",
            "remote_user",
            "request_header_user_agent__os__family",
            "request_method",
            "request_first_line",
            "request_url_fragment",
            "status",
            "request_header_user_agent",
            "remote_logname"
            ]

def prepare_data():
    #datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetNorm = dataset.sample(frac=1)
    datasetTrain = datasetNorm[dataset.index < train_length]
    datasetTest = datasetNorm[dataset.index >= train_length]

    xTrain = datasetTrain[features].as_matrix()
    yTrain = datasetTrain[["label"]].as_matrix()
    xTest = datasetTest[features].as_matrix()
    yTest = datasetTest[["label"]].as_matrix()

    test_ids = datasetTest[["id"]].as_matrix()

    return xTrain, yTrain, xTest, yTest, test_ids


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


def show_log_lines(test_ids, pred_list):
    test_lines = []

    with open(log_file_path, "r") as f:
        log_lines = f.readlines()

    for idx, line_num in enumerate(test_ids):
        test_lines.append((log_lines[int(line_num)].replace("\n", ""), pred_list[idx]))

    for line in test_lines:
        print("{}    pred: {}".format(line[0], line[1]))

if __name__ == "__main__":
    loss, train_step, prediction, last_label, last_state, batchX_placeholder, batchY_placeholder = model()
    xTrain, yTrain, xTest, yTest, test_ids = prepare_data()
    pred_list = train(  loss, train_step, prediction, last_label,
                        last_state, batchX_placeholder, batchY_placeholder,
                        xTrain, yTrain, xTest, yTest)

    show_log_lines(test_ids, pred_list)
