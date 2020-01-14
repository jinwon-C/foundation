import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import foundation as fd
from sklearn.model_selection import KFold

batch_size = 64
num_freq = 10
num_data = 510
num_total_data = num_freq * num_data
num_kfold = 5

filter_size = [3, 3]

directory = '../data/20191208-3d/log10(abs)+imag/win5000/'

data, num_class = fd.read_data(directory)

kf = KFold(n_splits = num_kfold)
kf.get_n_splits(data)
data = np.array(data)

for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]
    
    train_data = fd.array_to_list(train_data)
    test_data = fd.array_to_list(test_data)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for d in train_data:
        train_x.append(d[0:num_total_data])
        train_y.append(fd.one_hot(int(d[-1]), num_class))

    for d in test_data:
        test_x.append(d[0:num_total_data])
        test_y.append(fd.one_hot(int(d[-1]), num_class))

    with tf.device('/gpu:2'):
        x = tf.placeholder(tf.float32, [None, num_total_data])
        y = tf.placeholder(tf.float32, [None, num_class])

        W_conv1 = fd.weight_variable(filter_size, 1, 4)
        b_conv1 = fd.bias_variable([4])
        x_image = tf.reshape(x, [-1, num_freq, num_data, 1])
        h_conv1 = tf.nn.relu(fd.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = fd.max_pool(h_conv1, 1, 4)    #(input_shape, filter_height, filter_width)

        W_conv2 = fd.weight_variable(filter_size, 4, 8)
        b_conv2 = fd.bias_variable([8])
        h_conv2 = tf.nn.relu(fd.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = fd.max_pool(h_conv2, 1, 4)

        W_conv3 = fd.weight_variable(filter_size, 8, 16)
        b_conv3 = fd.bias_variable([16])
        h_conv3 = tf.nn.relu(fd.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = fd.max_pool(h_conv3, 1, 2)

        W_conv4 = fd.weight_variable(filter_size, 16, 32)
        b_conv4 = fd.bias_variable([32])
        h_conv4 = tf.nn.relu(fd.conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = fd.max_pool(h_conv4, 1, 2)

        keep_prob = tf.placeholder(tf.float32)
        #rate = 1 - keep_prob
        #dr = tf.nn.dropout(h_pool4, rate)

        W_conv5 = fd.weight_variable(filter_size, 32, 64)
        b_conv5 = fd.bias_variable([64])
        h_conv5 = tf.nn.relu(fd.conv2d(h_conv4, W_conv5) + b_conv5)
        h_pool5 = fd.max_pool(h_conv5, 2, 2)

        W_conv6 = fd.weight_variable(filter_size, 64, 128)
        b_conv6 = fd.bias_variable([128])
        h_conv6 = tf.nn.relu(fd.conv2d(h_pool5, W_conv6) + b_conv6)
        h_pool6 = fd.max_pool(h_conv6, 2, 2)

        W_conv7 = fd.weight_variable(filter_size, 128, 256)
        b_conv7 = fd.bias_variable([256])
        h_conv7 = tf.nn.relu(fd.conv2d(h_pool6, W_conv7) + b_conv7)
        h_pool7 = fd.max_pool(h_conv7, 2, 2)

        W_conv8 = fd.weight_variable(filter_size, 256, 512)
        b_conv8 = fd.bias_variable([512])
        h_conv8 = tf.nn.relu(fd.conv2d(h_pool7, W_conv8) + b_conv8)
        h_pool8 = fd.max_pool(h_conv8, 2, 2)

        W_conv9 = fd.weight_variable(filter_size, 512, 1024)
        b_conv9 = fd.bias_variable([1024])
        h_conv9 = tf.nn.relu(fd.conv2d(h_pool8, W_conv9) + b_conv9)
        h_pool9 = fd.max_pool(h_conv9, 2, 2)

        flat = tf.reshape(h_pool9, [-1, 1024])

        W_fc = fd.weight_variable_fc([1024, num_class])
        b_fc = fd.bias_variable([num_class])
        y_conv = tf.nn.softmax(tf.matmul(flat, W_fc) + b_fc)

        cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_x = np.array(train_x).reshape(len(train_x), num_total_data)
    train_y = np.array(train_y).reshape(len(train_y), num_class)

    for j in range(20000):
        batch_x, batch_y = fd.get_batch(batch_size, train_x, train_y)
        train_step.run(session=sess, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
        if j%100==0:
            train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y, keep_prob:1.0})
            print('step', j, 'accuracy : ', train_accuracy)
            print('test accuracy : ', sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0}))
