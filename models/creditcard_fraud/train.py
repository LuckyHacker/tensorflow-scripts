import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout


train_size = 0.9
batch_size = 10
state_size = 64
num_features = 30
num_classes = 1
dropout = 0.5
learning_rate = 0.001
epochs = 5

csv_file_path = "data/creditcard_sampled.csv"

dataset = pd.read_csv(csv_file_path)
train_length = int(len(dataset.index) * train_size)
test_length = int(len(dataset.index) * (1.0 - train_size))

features = [
            "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
            "V10","V11","V12","V13","V14","V15","V16","V17","V18",
            "V19","V20","V21","V22","V23","V24","V25","V26","V27",
            "V28","Amount",
            ]

def prepare_data():
    dataset_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    dataset_norm = dataset_norm.sample(frac=1)
    dataset_train = dataset_norm[dataset.index < train_length]
    dataset_test = dataset_norm[dataset.index >= train_length]

    x_train = dataset_train[features].as_matrix()
    y_train = dataset_train[["Class"]].as_matrix()
    x_test = dataset_test[features].as_matrix()
    y_test = dataset_test[["Class"]].as_matrix()

    return x_train, y_train, x_test, y_test


def plot_data(x, y):
    bins = np.linspace(-2, 2, 10)

    plt.hist(x, bins, alpha=1, label='predicted: 0')
    plt.hist(y, bins, alpha=1, label='predicted: 1')
    plt.legend(loc='upper right')
    plt.show()


def train(x_train, y_train, x_test, y_test):
    # Reshape data
    x_train = x_train[:-(len(x_train) % num_features)].reshape([-1, num_features])
    y_train = y_train[:-(len(y_train) % num_features)].reshape([-1, num_classes])
    x_test = x_test[:-(len(x_test) % num_features)].reshape([-1, num_features])
    y_test = y_test[:-(len(y_test) % num_features)].reshape([-1, num_classes])

    # Network building
    model = Sequential()
    model.add(Dense(state_size, input_dim=num_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(state_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))


    preds = list(map(lambda x: int(round(x[0])), model.predict(x_test)))
    plot_data([0] * preds.count(0), [1] * preds.count(1))



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()
    train(x_train, y_train, x_test, y_test)
