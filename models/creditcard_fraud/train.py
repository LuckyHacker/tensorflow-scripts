import tflearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_size = 0.9
batch_size = 30
state_size = 12
num_features = 30
dropout = 0.2
learning_rate = 0.001
epochs = 100

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
    #datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    datasetNorm = dataset.sample(frac=1)
    datasetTrain = datasetNorm[dataset.index < train_length]
    datasetTest = datasetNorm[dataset.index >= train_length]

    xTrain = datasetTrain[features].as_matrix()
    yTrain = datasetTrain[["Class"]].as_matrix()
    xTest = datasetTest[features].as_matrix()
    yTest = datasetTest[["Class"]].as_matrix()

    return xTrain, yTrain, xTest, yTest

def plot_data(x, y):
    sns.set_style("darkgrid")
    plt.plot(x, y)
    plt.show()

def train(trainX, trainY, testX, testY):
    # Reshape data
    """
    trainX = trainX.reshape([len(trainX), num_features])
    testX = testX.reshape([len(testX), num_features])
    trainY = trainY.reshape([len(trainY), 1])
    testY = testY.reshape([len(testY), 1])
    """

    print(trainX.shape)
    print(trainY.shape)
    # Network building
    input_ = tflearn.input_data([None, num_features])
    print(input_)
    linear = tflearn.single_unit(input_)
    regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
    metric='R2', learning_rate=learning_rate)

    # Training
    model = tflearn.DNN(regression, tensorboard_verbose=3)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=batch_size, n_epoch=epochs)


if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest = prepare_data()
    train(xTrain, yTrain, xTest, yTest)
