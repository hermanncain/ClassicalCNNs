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
with slim.arg_scope([slim.conv2d],padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer(),# this is default in slim.conv2d
            weights_regularizer=slim.l2_regularizer(0.005)):

    # MNIST conv1: 28*28*1 -> 25*25*16 -> 23*23*16
    conv1 = slim.conv2d(x_image,16,[3,3],stride=1,scope='conv1')
    pool1 = slim.max_pool2d(conv1,[2,2],stride=1,scope='pool1')
    #lrn1 = tf.nn.lrn(pool1,2,1,1e-4,0.75,name='lrn1')

    # MNIST conv1: 23*23*16 -> 20*20*64 -> 18*18*64
    conv2 = slim.conv2d(pool1,64,[3,3],stride=1,scope='conv2')
    pool2 = slim.max_pool2d(conv2,[2,2],stride=1,scope='pool2')
    #lrn2 = tf.nn.lrn(pool2,2,1,1e-4,0.75,name='lrn2')

    # conv3: 18*18*64 -> 8*8*128
    conv3 = slim.conv2d(pool2,384,[2,2],stride=2,scope='conv3')

    # MNIST conv1: 8*8*128 -> 6*6*256 -> 4*4*256
    # no group because I only have 1 GPU
    conv4 = slim.conv2d(conv3,256,[2,2],stride=1,scope='conv4')
    pool4 = slim.max_pool2d(conv4,[2,2],stride=1,scope='pool4')

    # MNIST fc6: 4*4*256 -> 1*1*1024 -> 1024
    conv5 = slim.conv2d(pool4,1024,[4,4],stride=1,scope='conv5')
    fc = slim.flatten(conv5)

    # MNIST fc6: 1024 -> 1024
    fc1 = slim.fully_connected(fc,1024,scope='fc1')
    drop1 = slim.dropout(fc1,keep_prob)

    # MNIST fc7: 1024 -> 1024
    fc2 = slim.fully_connected(drop1,1024,scope='fc2')
    drop2 = slim.dropout(fc2,keep_prob)

    # fc8: 1024 -> 10
    y_conv = slim.fully_connected(drop2,10,scope='fc3')
    

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