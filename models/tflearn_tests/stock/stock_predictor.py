import tflearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

outfile = "prediction.png"
train_size = 0.9
batch_size = 100
num_neurons = 12
num_features = 4
dropout = 0.2
learning_rate = 0.001
epochs = 100

# Read csv and normalize
dataset = pd.read_csv('../../data/currency/ETHUSD_TechnicalIndicators.csv')
train_length = int(len(dataset.index) * train_size)
dataset_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
datasetTrain = dataset_norm[dataset.index < train_length]
datasetTest = dataset_norm[dataset.index >= train_length]

# csv to numpy array
trainX = datasetTrain[['Close','MACD','Stochastics','ATR']].as_matrix()
trainY = datasetTrain['CloseTarget'].as_matrix()
testX = datasetTest[['Close','MACD','Stochastics','ATR']].as_matrix()
testY = datasetTest['CloseTarget'].as_matrix()

# Reshape data
trainX = trainX.reshape([len(trainX), num_features])
testX = testX.reshape([len(testX), num_features])
trainY = trainY.reshape([len(trainY), 1])
testY = testY.reshape([len(testY), 1])

# Network building
net = tflearn.input_data([None, trainX.shape[1]])
net = tflearn.embedding(net, input_dim=trainX.shape[1], output_dim=num_neurons)

net = tflearn.lstm( net, n_units=num_neurons, return_seq=True,
                    activation="relu6", dropout=dropout)
net = tflearn.lstm( net, n_units=num_neurons, return_seq=True,
                    dropout=dropout)
net = tflearn.lstm( net, n_units=num_neurons, return_seq=False,
                    dropout=dropout)

net = tflearn.fully_connected(net, 1)

net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='mean_square')


# Training
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size, n_epoch=epochs)



prediction = model.predict(testX)

plt.figure(figsize=(18,7))
plt.plot(testY, label='Price', color='blue', marker='o')
plt.plot(prediction, label='Predicted', color='red', marker='o')
plt.title('Stock prediction')
plt.legend(loc='upper left')
plt.savefig(outfile)
