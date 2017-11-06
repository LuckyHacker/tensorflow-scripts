from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


train_size = 0.8
batch_size = 30
state_size = 54
num_features = 27
num_classes = 1
dropout = 0.2
learning_rate = 0.001
epochs = 5000

csv_file_path = "data/access.csv"
log_file_path = "data/access.log"

dataset = pd.read_csv(csv_file_path)
train_length = int(len(dataset.index) * train_size)
test_length = int(len(dataset.index) * (1.0 - train_size))

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
    datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetNorm = dataset.sample(frac=1)
    datasetTrain = datasetNorm[dataset.index < train_length]
    datasetTest = datasetNorm[dataset.index >= train_length]

    xTrain = datasetTrain[features].as_matrix()
    yTrain = datasetTrain[["label"]].as_matrix()
    xTest = datasetTest[features].as_matrix()
    yTest = datasetTest[["label"]].as_matrix()

    test_ids = datasetTest[["id"]].as_matrix()

    return xTrain, yTrain, xTest, yTest, test_ids


def plot_data(x, y):
    bins = np.linspace(-2, 2, 10)

    plt.hist(x, bins, alpha=1, label='predicted: 0')
    plt.hist(y, bins, alpha=1, label='predicted: 1')
    plt.legend(loc='upper right')
    plt.show()


def train(x_train, y_train, x_test, y_test):
    # Network building
    model = Sequential()
    model.add(Dense(state_size, input_dim=num_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(state_size // 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(state_size // 4, activation='softmax'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    tensorboardlog = TensorBoard(log_dir='./keras_log', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboardlog])

    losses = history.history["loss"]
    plt.plot(list(range(len(losses))), losses, "-")
    plt.show()

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))


    preds = list(map(lambda x: int(round(x[0])), model.predict(x_test)))
    plot_data([0] * preds.count(0), [1] * preds.count(1))
    return preds


def show_log_lines(test_ids, pred_list):
    test_lines = []

    with open(log_file_path, "r") as f:
        log_lines = f.readlines()

    for idx, line_num in enumerate(test_ids):
        test_lines.append((log_lines[int(line_num)].replace("\n", ""), pred_list[idx]))

    for line in test_lines:
        print("{}    pred: {}".format(line[0], line[1]))

if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest, test_ids = prepare_data()
    pred_list = train(xTrain, yTrain, xTest, yTest)
    show_log_lines(test_ids, pred_list)
