import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

outfile = "prediction.png"
dataset = pd.read_csv('data/currency/ETHUSD_TechnicalIndicators.csv')
datasetNorm = (dataset - dataset.min()) / (dataset.max() - dataset.min())



num_epochs = 100
batch_size = 1
train_size = 0.9
total_series_length = int(len(dataset.index) * train_size)
truncated_backprop_length = 3 #The size of the sequence
state_size = 12 #The number of neurons
num_features = 4
num_classes = 1 #[1,0]
num_batches = total_series_length//batch_size//truncated_backprop_length
min_test_size = 100
print('The total series length is: %d' %total_series_length)
print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past'
      %(num_batches,batch_size,truncated_backprop_length))

datasetTrain = datasetNorm[dataset.index < total_series_length]
datasetTest = datasetNorm[dataset.index >= total_series_length]

xTrain = datasetTrain[['Price','MACD','Stochastics','ATR']].as_matrix()
yTrain = datasetTrain['PriceTarget'].as_matrix()
xTest = datasetTest[['Price','MACD','Stochastics','ATR']].as_matrix()
yTest = datasetTest['PriceTarget'].as_matrix()

batchX_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_features],name='data_ph')
batchY_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_classes],name='target_ph')
W2 = tf.Variable(initial_value=np.random.rand(state_size,num_classes),dtype=tf.float32)
b2 = tf.Variable(initial_value=np.random.rand(1,num_classes),dtype=tf.float32)

labels_series = tf.unstack(batchY_placeholder, axis=1)
cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)
states_series, current_state = tf.nn.dynamic_rnn(cell=cell,inputs=batchX_placeholder,dtype=tf.float32)
states_series = tf.transpose(states_series,[1,0,2])

last_state = tf.gather(params=states_series,indices=states_series.get_shape()[0]-1)
last_label = tf.gather(params=labels_series,indices=len(labels_series)-1)

weight = tf.Variable(tf.truncated_normal([state_size,num_classes]))
bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))

prediction = tf.matmul(last_state,weight) + bias

loss = tf.reduce_mean(tf.squared_difference(last_label,prediction))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)



loss_list = []
test_pred_list = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch_idx in range(num_epochs):
        print('Epoch %d' %epoch_idx)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length * batch_size


            batchX = xTrain[start_idx:end_idx,:].reshape(batch_size,truncated_backprop_length,num_features)
            batchY = yTrain[start_idx:end_idx].reshape(batch_size,truncated_backprop_length,1)

            #print('IDXs',start_idx,end_idx)
            #print('X',batchX.shape,batchX)
            #print('Y',batchX.shape,batchY)

            feed = {batchX_placeholder : batchX, batchY_placeholder : batchY}

            #TRAIN!
            _loss,_train_step,_pred,_last_label,_prediction = sess.run(
                fetches=[loss, train_step, prediction, last_label, prediction],
                feed_dict = feed
            )

            loss_list.append(_loss)
            if(batch_idx % 200 == 0):
                print('Step %d - Loss: %.6f' %(batch_idx,_loss))

    #TEST
    for test_idx in range(len(xTest) - truncated_backprop_length):

        testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1, truncated_backprop_length, num_features))
        testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1, truncated_backprop_length, 1))

        #_current_state = np.zeros((batch_size,state_size))
        feed = {batchX_placeholder : testBatchX,
            batchY_placeholder : testBatchY}

        #Test_pred contains 'window_size' predictions, we want the last one
        _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
        test_pred_list.append(test_pred[-1][0]) #The last one


plt.figure(figsize=(18,7))
plt.plot(yTest,label='Price',color='blue')
plt.plot(test_pred_list,label='Predicted',color='red')
plt.title('Price vs Predicted')
plt.legend(loc='upper left')
plt.savefig(outfile)
