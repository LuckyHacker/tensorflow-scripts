# tutorial checkpoint: https://www.tensorflow.org/get_started/mnist/pros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

image_size = (28, 28)
image_vector_size = 28 * 28
num_classes = 10
batch_size = 100
steps = 1000
learning_rate = 0.5

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, image_vector_size])
y_ = tf.placeholder(tf.float32, [None, num_classes])

W = tf.Variable(tf.zeros([image_vector_size, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

for _ in range(steps):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
