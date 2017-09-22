import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


 # Read and normalize data
outfile = "prediction2.png"
dataset = pd.read_csv('../data/currency/EURUSD_TechnicalIndicators.csv')
dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())


# Hyperparams
num_epochs = 100
batch_size = 1
total_series_length = len(dataset.index)
truncated_backprop_length = 3 #The size of the sequence
state_size = 12 #The number of neurons

num_features = 4
num_classes = 1 #[1,0]
num_batches = total_series_length//batch_size//truncated_backprop_length
min_test_size = 100


def train_one_classifier(num_epochs,label_n):
    display_freq = num_epochs//10

    loss_list = []
    loss_list_batch = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in range(num_epochs):
            print('Step: %d' %step)

            for batch_idx in range(num_batches):

                batchx = xTrain[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE,:]
                batchy = yTrain[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE].ravel()

                feed = {x: batchx, y: batchy, keep_prob:0.5}

                _,_loss = sess.run([optimizer,cost], feed_dict=feed)
                loss_list_batch.append(np.average(_loss)) #Store av. batch loss

            #Add the av. loss of the batch to the loss list
            loss_list.append(np.average(loss_list_batch))
            loss_list_batch = []

            if(step == 1):
                print('Initial loss: %.3f' % np.average(_loss))



        feed_dev = {x : xTest, y : yTest.ravel(), keep_prob:1}
        _y,_softmax,_pred = sess.run([y,logits,pred],feed_dict=feed_dev)

        print('Train finished - Final loss: %.3f' % (np.average(_loss)))


    return loss_list,_y,_pred


print('The total series length is: %d' %total_series_length)
print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past'
      %(num_batches,batch_size,truncated_backprop_length))


# Train-Test split
datasetTrain = dataset[dataset.index < num_batches*batch_size*truncated_backprop_length]

for i in range(min_test_size,len(dataset.index)):

    if(i % truncated_backprop_length*batch_size == 0):
        test_first_idx = len(dataset.index)-i
        break

datasetTest =  dataset[dataset.index >= test_first_idx]


xTrain = datasetTrain[['Close','MACD','Stochastics','ATR']].as_matrix()
yTrain = datasetTrain['CloseTarget'].as_matrix()
xTest = datasetTest[['Close','MACD','Stochastics','ATR']].as_matrix()
yTest = datasetTest['CloseTarget'].as_matrix()



LEARNING_RATE = 0.001
BATCH_SIZE = 1

x = tf.placeholder(tf.float32,[None,num_features],name='x')
y = tf.placeholder(tf.float32,[None],name='y')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')

#Layer1
WRelu1 = tf.Variable(tf.truncated_normal([num_features,num_features]),dtype=tf.float32,name='wrelu1')
bRelu1 = tf.Variable(tf.truncated_normal([num_features]),dtype=tf.float32,name='brelu1')
relu1 = tf.nn.elu(tf.matmul(x,WRelu1) + bRelu1,name='relu1')

#DROPOUT
relu1 = tf.nn.dropout(relu1,keep_prob=keep_prob,name='relu1drop')

#Layer2
WRelu2 = tf.Variable(tf.truncated_normal([num_features,num_features]),dtype=tf.float32,name='wrelu2')
bRelu2 = tf.Variable(tf.truncated_normal([num_features]),dtype=tf.float32,name='brelu2')
layer2 = tf.add(tf.add(tf.matmul(relu1,WRelu2),bRelu2),x,name='layer2')

relu2 = tf.nn.elu(layer2,name='relu2')

#Layer3
WRelu3 = tf.Variable(tf.truncated_normal([num_features,num_features]),dtype=tf.float32,name='wrelu3')
bRelu3 = tf.Variable(tf.truncated_normal([num_features]),dtype=tf.float32,name='brelu3')
relu3 = tf.nn.elu(tf.matmul(relu2,WRelu3) + bRelu3,name='relu3')


#DROPOUT
relu3 = tf.nn.dropout(relu3,keep_prob=keep_prob,name='relu3drop')

#Out layer
Wout = tf.Variable(tf.truncated_normal([num_features,1]),dtype=tf.float32,name='wout')
bout = tf.Variable(tf.truncated_normal([1]),dtype=tf.float32,name='bout')
logits = tf.add(tf.matmul(relu3,Wout),bout,name='logits')


#Predictions
pred = logits

#Cost & Optimizer
cost = tf.reduce_mean(tf.squared_difference(y, logits))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


loss_list,_y,_pred = train_one_classifier(num_epochs,1)

plt.figure(figsize=(21,7))
plt.plot(_y,label='Price',color='blue')
plt.plot(_pred.ravel(),label='Predicted',color='red')
plt.title('Price vs Predicted')
plt.legend(loc='upper left')
plt.savefig(outfile)
