import tflearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

outfile = "prediction.png"
train_size = 0.9
batch_size = 1
num_neurons = 128
input_dim = 1000
num_features = 4
dropout = 0.9
learning_rate = 0.001
epochs = 5

# Read csv and normalize
dataset = pd.read_csv('../../data/currency/ETHUSD_TechnicalIndicators.csv')
train_length = int(len(dataset.index) * train_size)
dataset_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
datasetTrain = dataset_norm[dataset.index < train_length]
datasetTest = dataset_norm[dataset.index >= train_length]

# csv to numpy array
xTrain = datasetTrain[['Close','MACD','Stochastics','ATR']].as_matrix()
yTrain = datasetTrain['CloseTarget'].as_matrix()
xTest = datasetTest[['Close','MACD','Stochastics','ATR']].as_matrix()
yTest = datasetTest['CloseTarget'].as_matrix()

# Reshape data
xTrain = xTrain.reshape([len(xTrain), num_features])
xTest = xTest.reshape([len(xTest), num_features])
yTrain = yTrain.reshape([len(yTrain), 1])
yTest = yTest.reshape([len(yTest), 1])

# Network building
net = tflearn.input_data([None, 4])

net = tflearn.embedding(net, input_dim=num_features, output_dim=num_neurons)
net = tflearn.lstm(net, n_units=num_neurons, return_seq=True)
net = tflearn.lstm(net, n_units=num_neurons, return_seq=True)
net = tflearn.lstm(net, n_units=num_neurons, return_seq=False)
net = tflearn.fully_connected(net, 1, activation='relu')

net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='mean_square')


# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(xTrain, yTrain, validation_set=(xTest, yTest), show_metric=True,
          batch_size=batch_size, n_epoch=epochs)



prediction = model.predict(xTest)

plt.figure(figsize=(18,7))
plt.plot(yTest, label='Price', color='blue', marker='o')
plt.plot(prediction, label='Predicted', color='red', marker='o')
plt.title('Stock prediction')
plt.legend(loc='upper left')
plt.savefig(outfile)
