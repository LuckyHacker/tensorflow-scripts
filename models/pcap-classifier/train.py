from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from collections import Counter
from dataset_tools import normalize, SplitDataframe, Balancer, tokenize_dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json


train_size = 0.8
batch_size = 2048
state_size = 90
num_features = 43
num_classes = 1
dropout = 0.5
learning_rate = 0.001
epochs = 500

train_csv_file_path = "data/UNSW_NB15_training-set.csv"
test_csv_file_path = "data/UNSW_NB15_testing-set.csv"

df_train = pd.read_csv(train_csv_file_path)
df_test = pd.read_csv(test_csv_file_path)
df = pd.concat([df_train, df_test], ignore_index=False)


def count_dict(l):
    return Counter(l)


def prepare_data(df):
    df = tokenize_dataframe(df, "tokens")
    dataset_norm = normalize(df)
    splitter = SplitDataframe(dataset_norm, train_size=train_size)
    dataset_train = splitter.traindata
    dataset_test = splitter.testdata

    x_train = dataset_train.drop(["label", "attack_cat"], axis=1).as_matrix()
    y_train = dataset_train[["label"]].as_matrix()
    x_test = dataset_test.drop(["label", "attack_cat"], axis=1).as_matrix()
    y_test = dataset_test[["label"]].as_matrix()

    return x_train, y_train, x_test, y_test


def plot_data(d):
    print(d)
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), d.keys())
    plt.legend(loc='upper right')
    plt.show()


def train(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(state_size, input_dim=num_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    tensorboardlog = TensorBoard(log_dir='./keras_log', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboardlog])

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))

    losses = history.history["loss"]
    plt.plot(list(range(len(losses))), losses, "-")
    plt.show()

    preds = list(map(lambda x: int(round(x[0])), model.predict(x_test)))

    count_labels = count_dict(preds)
    plot_data(count_labels)
    return preds

if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest = prepare_data(df)
    pred_list = train(xTrain, yTrain, xTest, yTest)
