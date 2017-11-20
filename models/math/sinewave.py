import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import TensorBoard


train_size = 0.8
batch_size = 512
state_size = 2
num_features = 1
num_classes = 1
dropout = 0.5
learning_rate = 0.001
epochs = 1

"""
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()
"""


def train_test_split_list(l, ratio):
    return (l[:int(len(l) * ratio)], l[int(len(l) * ratio):])


def prepare_data():
    Fs = 8000
    f = 5
    r = np.arange(100000)
    x = np.sin(2 * np.pi * f * r / Fs)[:-1]
    y = np.sin(2 * np.pi * f * r / Fs)[1:]

    x_train, x_test = train_test_split_list(x, 0.95)
    y_train, y_test = train_test_split_list(y, 0.95)

    """
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    """

    """
    x_train = x_train[:-(len(x_train) % num_features)].reshape([-1, num_features])
    y_train = y_train[:-(len(y_train) % num_features)].reshape([-1, num_classes])
    x_test = x_test[:-(len(x_test) % num_features)].reshape([-1, num_features])
    y_test = y_test[:-(len(y_test) % num_features)].reshape([-1, num_classes])
    """

    return x_train, y_train, x_test, y_test


def plot_data(x, y):
    bins = np.linspace(-2, 2, 10)

    plt.hist(x, bins, alpha=1, label='predicted: 0')
    plt.hist(y, bins, alpha=1, label='predicted: 1')
    plt.legend(loc='upper right')
    plt.show()


def train(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(num_features, state_size))
    model.add(LSTM(state_size))
    model.add(Dropout(dropout))
    model.add(Dense(num_features, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))

    preds = list(map(lambda x: model.predict(x), np.array(x_test)))
    plot_data(preds, list(range(len(preds))))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()
    train(x_train, y_train, x_test, y_test)
