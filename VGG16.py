import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
import random

image_size = 32
img_channels = 3
lr = 0.00001
keep_prob = tf.placeholder(tf.float32)
epoch = 100
batch_size=50
sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, image_size, image_size, img_channels])
y = tf.placeholder("float", [None, 10])

# data loading
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def prepare_data():
    print("======Loading data======")
    data_dir = './cifar10'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

# data augmentation
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

def data_preprocessing(x_train,x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch

def vgg16(inputs):
    with slim.arg_scope([slim.conv2d],kernel_size=[3,3],padding='SAME',stride=1):
        # slim.max_pool2d default stride=2
        with slim.arg_scope([slim.max_pool2d],kernel_size=[2,2]):
            net = slim.repeat(inputs,2,slim.conv2d,64,scope='conv1')
            net = slim.max_pool2d(net,scope='pool1')
            net = slim.repeat(net,2,slim.conv2d,128,scope='conv2')
            net = slim.max_pool2d(net,scope='pool2')
            net = slim.repeat(net,3,slim.conv2d,256,scope='conv3')
            net = slim.max_pool2d(net,scope='pool3')
            net = slim.repeat(net,3,slim.conv2d,512,scope='conv4')
            net = slim.max_pool2d(net,scope='pool4')
            net = slim.repeat(net,3,slim.conv2d,512,scope='conv5')
            net = slim.max_pool2d(net,scope='pool5')
            net = slim.flatten(net)
            net = slim.fully_connected(net,4096,scope='fc1')
            net = slim.dropout(net,keep_prob)
            net = slim.fully_connected(net,4096,scope='fc2')
            net = slim.dropout(net,keep_prob)
            # modified 1000 to 10 for cifar-10 dataset
            net = slim.fully_connected(net,10,scope='fc3')
    return net

# graph
logits = vgg16(x)
loss = slim.losses.softmax_cross_entropy(logits,y)

opt = tf.train.AdamOptimizer(lr).minimize(loss)#.GradientDescentOptimizer(lr).minimize(loss)

is_correct = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(is_correct,tf.float32))

# load data
train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = data_preprocessing(train_x, test_x)
print(test_y[0])
# train
tf.global_variables_initializer().run()
batch_idx = int(len(train_x)//batch_size)
for i in range(epoch):
    for j in range(batch_idx):
        batch_x = train_x[j*batch_size:(j+1)*batch_size]
        batch_y = train_y[j*batch_size:(j+1)*batch_size]
        batch_x = data_augmentation(batch_x)
        if j % 50 == 0:
            train_acc = acc.eval(feed_dict={x:batch_x,y:batch_y,keep_prob:1})
            train_loss = loss.eval(feed_dict={x:batch_x,y:batch_y,keep_prob:1})
            #test_acc = acc.eval(feed_dict={x:test_x,y:test_y,keep_prob:1})
            #print("step %d, train_acc: %g, test_acc: %g"%(i,train_acc,test_acc))
            print("epoch %d, batch %d, train_loss:%g, train_acc: %g"%(i,j,train_loss,train_acc))
        opt.run(feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})