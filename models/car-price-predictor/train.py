from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import TensorBoard
from collections import Counter
from dataset_tools import normalize, SplitDataframe, Balancer, tokenize_dataframe, DeNormalizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


train_size = 0.8
batch_size = 1024
state_size = 14
num_features = 7
num_classes = 1
dropout = 0.5
learning_rate = 0.001
epochs = 10

model_path = "model"
csv_file_path = "data/86k_cars_finland.csv"

df = pd.read_csv(csv_file_path)


def count_dict(l):
    return Counter(l)


def preprocess_dataframe(df):
    df = df.drop(["desc"], axis=1)
    df["kilometers"] = list(map(lambda x: "0" if x == "Ajamaton" else x, df["kilometers"]))
    df["kilometers"] = list(map(lambda x: x.replace(" ", ""), df["kilometers"]))
    df["kilometers"] = list(map(lambda x: int(x.replace("km", "")), df["kilometers"]))
    df["price"] = list(map(lambda x: int(x.replace("€", "")) if x != "Eihinnoiteltu" else np.NaN, df["price"]))
    df["fuel"] = list(map(lambda x: x if "km" not in str(x) else np.NaN, df["fuel"]))
    df["fuel"] = list(map(lambda x: x if "nan" not in str(x) else np.NaN, df["fuel"]))
    df["fuel"] = list(map(lambda x: x if "Ajamaton" not in str(x) else np.NaN, df["fuel"]))
    df["transmission"] = list(map(lambda x: x if "Manuaali" in x or "Automaatti" in x else np.NaN, df["transmission"]))
    df = df.dropna()
    return df

def prepare_data(df):
    df = tokenize_dataframe(df, "tokens")
    dataset_norm = normalize(df)
    splitter = SplitDataframe(dataset_norm, train_size=train_size)
    dataset_train = splitter.traindata
    dataset_test = splitter.testdata

    x_train = dataset_train.drop(["price"], axis=1).as_matrix()
    y_train = dataset_train[["price"]].as_matrix()
    x_test = dataset_test.drop(["price"], axis=1).as_matrix()
    y_test = dataset_test[["price"]].as_matrix()

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
    model.add(Dense(state_size // 2, input_dim=num_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(state_size // 4, input_dim=num_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    tensorboardlog = TensorBoard(log_dir='./keras_log', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboardlog])

    with open("{}.json".format(model_path), "w") as f:
        f.write(model.to_json())

    model.save_weights("{}.h5".format(model_path))

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nAccuracy: {0:.2f} % ".format(score * 100))

    losses = history.history["loss"]
    plt.plot(list(range(len(losses))), losses, "-")
    plt.show()


    denorm = DeNormalizer(df, "price")
    y_test = list(map(lambda x: denorm.convert_scalar(x[0]), y_test))
    preds = list(map(lambda x: denorm.convert_scalar(x[0]), model.predict(x_test)))
    preds_label = []
    for x, y in list(zip(preds, y_test)):
        diff = y / 100 * 20
        if abs(x - y) < diff:
            preds_label.append(1)
        else:
            preds_label.append(0)

    count_labels = count_dict(preds_label)
    print(count_labels[1] / (count_labels[1] + count_labels[0]) * 100)
    return preds

if __name__ == "__main__":
    df = preprocess_dataframe(df)
    xTrain, yTrain, xTest, yTest = prepare_data(df)
    pred_list = train(xTrain, yTrain, xTest, yTest)
