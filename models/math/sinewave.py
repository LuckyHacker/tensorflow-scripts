import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation
from keras.callbacks import TensorBoard


train_size = 0.8
batch_size = 64
state_size = 100
num_features = 1
num_classes = 1
dropout = 0.2
learning_rate = 0.001
epochs = 3


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
    x_train = x_train[:-(len(x_train) % num_features)].reshape([-1, num_features])
    y_train = y_train[:-(len(y_train) % num_features)].reshape([-1, num_classes])
    x_test = x_test[:-(len(x_test) % num_features)].reshape([-1, num_features])
    y_test = y_test[:-(len(y_test) % num_features)].reshape([-1, num_classes])
    """

    print(x_train)
    input()

    return x_train, y_train, x_test, y_test


def plot_data(x, y):
    plt.plot(x, y)
    plt.show()


def train(x_train, y_train, x_test, y_test):
    layers = [1, 50, 100, 1]
    model = Sequential()
    model.add(Embedding(input_dim=layers[0], output_dim=layers[1]))
    model.add(LSTM(state_size))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))

    preds = list(map(lambda x: x[0], model.predict(x_test)))
    plot_data(list(range(len(preds))), preds)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()
    train(x_train, y_train, x_test, y_test)
