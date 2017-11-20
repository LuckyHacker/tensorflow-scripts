import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import TensorBoard
from itertools import count
from collections import defaultdict


train_size = 0.8
batch_size = 64
state_size = 256
num_features = 100
num_classes = 100
dropout = 0.5
learning_rate = 0.001
epochs = 50

data_path = "data/"


class TokenizeWords:

    def __init__(self):
        self.id_dict = {}
        self.count_id = 1

    def tokenize_list(self, l):
        words = l.split(" ")
        token_list = []
        for item in words:
            if item in self.id_dict:
                token_list.append(self.id_dict[item])
            else:
                self.id_dict[item] = self.count_id
                self.count_id += 1

                token_list.append(self.id_dict[item])

        return token_list

    def get_id_dict(self):
        return self.id_dict

    def id_dict_to_json(self, path):
        json_string = json.dumps(self.id_dict, sort_keys=True, indent=4)
        with open(path, "w") as f:
            f.write(json_string)


def train_test_split_list(l, ratio):
    return (l[:int(len(l) * ratio)], l[int(len(l) * ratio):])


def pad_list(l, req_len):
    for i in range(req_len - len(l)):
        l.append(-1)

    l = l[:req_len]

    return l

def prepare_data():
    tokenizer = TokenizeWords()
    lines = []

    """
    for _file in os.listdir(data_path):
        with open("{}{}".format(data_path, _file)) as f:
            lines += f.readlines()
    """

    with open("data/m0") as f:
        lines = f.readlines()

    lines = list(map(lambda x: tokenizer.tokenize_list(x.strip()), lines))
    tokenizer.id_dict_to_json("tokens.json")

    lines = list(map(lambda x: np.array(pad_list(x, num_features)), lines))

    x = []
    for i in range(len(lines) - 1):
        x.append(lines[i])

    y = []
    for i in range(len(lines) - 1):
        i += 1
        y.append(lines[i])

    x_train, x_test = train_test_split_list(x, train_size)
    y_train, y_test = train_test_split_list(y, train_size)

    x_train = normalize(np.array(x_train))
    x_test = normalize(np.array(x_test))
    y_train = normalize(np.array(y_train))
    y_test = normalize(np.array(y_test))

    x_train = x_train[:-(len(x_train) % num_features)].reshape([-1, num_features])
    y_train = y_train[:-(len(y_train) % num_features)].reshape([-1, num_classes])
    x_test = x_test[:-(len(x_test) % num_features)].reshape([-1, num_features])
    y_test = y_test[:-(len(y_test) % num_features)].reshape([-1, num_classes])

    return x_train, y_train, x_test, y_test


def normalize(l):
    row_sums = l.sum(axis=1)
    new_matrix = l / row_sums[:, np.newaxis]
    return new_matrix


def plot_data(x, y):
    bins = np.linspace(-2, 2, 10)

    plt.hist(x, bins, alpha=1, label='predicted: 0')
    plt.hist(y, bins, alpha=1, label='predicted: 1')
    plt.legend(loc='upper right')
    plt.show()


def untokenize_sentence(l):
    with open("tokens.json", "r") as f:
        data = json.loads(f.read())

    data = {v: k for k, v in data.items()}
    sentence = []
    for word in l:
        sentence.append(data[word])

    return " ".join(sentence)


def train(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(num_features, state_size))
    model.add(LSTM(state_size // 2))
    model.add(Dropout(dropout))
    model.add(Dense(num_features, activation="sigmoid"))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    tensorboardlog = TensorBoard(log_dir='./keras_log', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboardlog])

    losses = history.history["loss"]
    plt.plot(list(range(len(losses))), losses, "-")
    plt.show()

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))

    sentence = model.predict(np.array(pad_list([29397, 4743], num_features)).reshape([-1, num_features]))

    sentence = list(map(lambda x: round(x), sentence[0]))
    print(untokenize_sentence(sentence))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()
    train(x_train, y_train, x_test, y_test)
