import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
sess = tf.InteractiveSession()
 
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

# LeNet
# replaced sigmoid with ReLU
# add dropout
keep_prob = tf.placeholder(tf.float32)

# Conv1 Layer
with slim.arg_scope([slim.conv2d],padding='VALID',
            #weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.005)):
    # C1: 1*32*32 -> 6*28*28
    # but mnist image is 28*28, so use padding=same instead
    h1 = slim.conv2d(x_image,6,[5,5],padding='SAME',scope='conv1')
    # S2: 6*28*28 -> 6*14*14
    h2 = slim.max_pool2d(h1,[2,2],scope='pool2')
    # C3: 6*14*14 -> 16*10*10
    h3 = slim.conv2d(h2,16,[5,5],scope='conv3')
    # S4: 16*10*10 -> 16*5*5
    h4 = slim.max_pool2d(h3,[2,2],scope='pool4')
    # C5: 16*5*5 -> 120*1*1
    h5 = slim.conv2d(h4,120,[5,5],scope='conv5')
    h5 = slim.flatten(h5)
    # F6: 120 -> 84
    # because ASCII codes are 7*12=84 pixels
    h6 = slim.fully_connected(h5,84,scope='fc6')
    h6 = slim.dropout(h6, keep_prob)
    # OUTPUT: 84 -> 10
    # simply replace gaussian connection(rbf) with orinary fc
    # because softmax has better possibility property
    # and softmax_cross_entropy will be used as loss
    y_conv = slim.fully_connected(h6,10,scope='fc7')

loss = slim.losses.softmax_cross_entropy(y_conv, y)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
tf.global_variables_initializer().run()
for i in range(5000):
	batch = mnist.train.next_batch(50)
	if i % 500 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
		print("step %d, acc: %g"%(i,train_accuracy))
	train_step.run(feed_dict={x:batch[0],y:batch[1],keep_prob:0.75})
 
print("test acc: %g"%accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))
