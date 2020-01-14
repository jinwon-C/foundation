import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import dual_foundation as fd
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
        train_x.append(d[0:2 * num_total_data])
        train_y.append(fd.one_hot(int(d[-1]), num_class))

    for d in test_data:
        test_x.append(d[0:2 * num_total_data])
        test_y.append(fd.one_hot(int(d[-1]), num_class))

    with tf.device('/gpu:2'):
        x1 = tf.placeholder(tf.float32, [None, num_total_data])
        y1 = tf.placeholder(tf.float32, [None, num_class])

        W_conv11 = fd.weight_variable(filter_size, 1, 4)
        b_conv11 = fd.bias_variable([4])
        x_image1 = tf.reshape(x1, [-1, num_freq, num_data, 1])
        h_conv11 = tf.nn.relu(fd.conv2d(x_image1, W_conv11) + b_conv11)
        h_pool11 = fd.max_pool(h_conv11, 1, 4)    #(input_shape, filter_height, filter_width)

        W_conv12 = fd.weight_variable(filter_size, 4, 8)
        b_conv12 = fd.bias_variable([8])
        h_conv12 = tf.nn.relu(fd.conv2d(h_pool11, W_conv12) + b_conv12)
        h_pool12 = fd.max_pool(h_conv12, 1, 4)

        W_conv13 = fd.weight_variable(filter_size, 8, 16)
        b_conv13 = fd.bias_variable([16])
        h_conv13 = tf.nn.relu(fd.conv2d(h_pool12, W_conv13) + b_conv13)
        h_pool13 = fd.max_pool(h_conv13, 1, 2)

        W_conv14 = fd.weight_variable(filter_size, 16, 32)
        b_conv14 = fd.bias_variable([32])
        h_conv14 = tf.nn.relu(fd.conv2d(h_pool13, W_conv14) + b_conv14)
        h_pool14 = fd.max_pool(h_conv14, 1, 2)

        keep_prob = tf.placeholder(tf.float32)
        #rate = 1 - keep_prob
        #dr = tf.nn.dropout(h_pool4, rate)

        W_conv15 = fd.weight_variable(filter_size, 32, 64)
        b_conv15 = fd.bias_variable([64])
        h_conv15 = tf.nn.relu(fd.conv2d(h_conv14, W_conv15) + b_conv15)
        h_pool15 = fd.max_pool(h_conv15, 2, 2)

        W_conv16 = fd.weight_variable(filter_size, 64, 128)
        b_conv16 = fd.bias_variable([128])
        h_conv16 = tf.nn.relu(fd.conv2d(h_pool15, W_conv16) + b_conv16)
        h_pool16 = fd.max_pool(h_conv16, 2, 2)

        W_conv17 = fd.weight_variable(filter_size, 128, 256)
        b_conv17 = fd.bias_variable([256])
        h_conv17 = tf.nn.relu(fd.conv2d(h_pool16, W_conv17) + b_conv17)
        h_pool17 = fd.max_pool(h_conv17, 2, 2)

        W_conv18 = fd.weight_variable(filter_size, 256, 512)
        b_conv18 = fd.bias_variable([512])
        h_conv18 = tf.nn.relu(fd.conv2d(h_pool17, W_conv18) + b_conv18)
        h_pool18 = fd.max_pool(h_conv18, 2, 2)

        W_conv19 = fd.weight_variable(filter_size, 512, 1024)
        b_conv19 = fd.bias_variable([1024])
        h_conv19 = tf.nn.relu(fd.conv2d(h_pool18, W_conv19) + b_conv19)
        h_pool19 = fd.max_pool(h_conv19, 2, 2)

        flat1 = tf.reshape(h_pool19, [-1, 1024])

        W_fc1 = fd.weight_variable_fc([1024, num_class])
        b_fc1 = fd.bias_variable([num_class])
        y_conv1 = tf.nn.softmax(tf.matmul(flat1, W_fc1) + b_fc1)

        cross_entropy1 = -tf.reduce_sum(y1 * tf.log(tf.clip_by_value(y_conv1, 1e-10, 1.0)))
        train_step1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy1)
        correct_prediction1 = tf.equal(tf.argmax(y_conv1, 1), tf.argmax(y1, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))


        x2 = tf.placeholder(tf.float32, [None, num_total_data])
        y2 = tf.placeholder(tf.float32, [None, num_class])

        W_conv21 = fd.weight_variable(filter_size, 1, 4)
        b_conv21 = fd.bias_variable([4])
        x_image2 = tf.reshape(x2, [-1, num_freq, num_data, 1])
        h_conv21 = tf.nn.relu(fd.conv2d(x_image2, W_conv21) + b_conv21)
        h_pool21 = fd.max_pool(h_conv21, 1, 4)    #(input_shape, filter_height, filter_width)

        W_conv22 = fd.weight_variable(filter_size, 4, 8)
        b_conv22 = fd.bias_variable([8])
        h_conv22 = tf.nn.relu(fd.conv2d(h_pool21, W_conv22) + b_conv22)
        h_pool22 = fd.max_pool(h_conv22, 1, 4)

        W_conv23 = fd.weight_variable(filter_size, 8, 16)
        b_conv23 = fd.bias_variable([16])
        h_conv23 = tf.nn.relu(fd.conv2d(h_pool22, W_conv23) + b_conv23)
        h_pool23 = fd.max_pool(h_conv23, 1, 2)

        W_conv24 = fd.weight_variable(filter_size, 16, 32)
        b_conv24 = fd.bias_variable([32])
        h_conv24 = tf.nn.relu(fd.conv2d(h_pool23, W_conv24) + b_conv24)
        h_pool24 = fd.max_pool(h_conv24, 1, 2)

        keep_prob = tf.placeholder(tf.float32)
        #rate = 1 - keep_prob
        #dr = tf.nn.dropout(h_pool4, rate)

        W_conv25 = fd.weight_variable(filter_size, 32, 64)
        b_conv25 = fd.bias_variable([64])
        h_conv25 = tf.nn.relu(fd.conv2d(h_conv24, W_conv25) + b_conv25)
        h_pool25 = fd.max_pool(h_conv25, 2, 2)

        W_conv26 = fd.weight_variable(filter_size, 64, 128)
        b_conv26 = fd.bias_variable([128])
        h_conv26 = tf.nn.relu(fd.conv2d(h_pool25, W_conv26) + b_conv26)
        h_pool26 = fd.max_pool(h_conv26, 2, 2)

        W_conv27 = fd.weight_variable(filter_size, 128, 256)
        b_conv27 = fd.bias_variable([256])
        h_conv27 = tf.nn.relu(fd.conv2d(h_pool26, W_conv27) + b_conv27)
        h_pool27 = fd.max_pool(h_conv27, 2, 2)

        W_conv28 = fd.weight_variable(filter_size, 256, 512)
        b_conv28 = fd.bias_variable([512])
        h_conv28 = tf.nn.relu(fd.conv2d(h_pool27, W_conv28) + b_conv28)
        h_pool28 = fd.max_pool(h_conv28, 2, 2)

        W_conv29 = fd.weight_variable(filter_size, 512, 1024)
        b_conv29 = fd.bias_variable([1024])
        h_conv29 = tf.nn.relu(fd.conv2d(h_pool28, W_conv29) + b_conv29)
        h_pool29 = fd.max_pool(h_conv29, 2, 2)

        flat2 = tf.reshape(h_pool29, [-1, 1024])

        W_fc2 = fd.weight_variable_fc([1024, num_class])
        b_fc2 = fd.bias_variable([num_class])
        y_conv2 = tf.nn.softmax(tf.matmul(flat2, W_fc2) + b_fc2)

        cross_entropy2 = -tf.reduce_sum(y2 * tf.log(tf.clip_by_value(y_conv2, 1e-10, 1.0)))
        train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy2)
        correct_prediction2 = tf.equal(tf.argmax(y_conv2, 1), tf.argmax(y2, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_x = np.array(train_x).reshape(len(train_x), 2*num_total_data)
    train_y = np.array(train_y).reshape(len(train_y), num_class)

    for j in range(20000):
        batch_x, batch_y = fd.get_batch(batch_size, train_x, train_y)
        batch_abs = batch_x[:,0: num_total_data]
        batch_imag = batch_x[: num_total_data:2*num_total_data]
        train_step1.run(session=sess, feed_dict={x1:batch_abs, y1:batch_y, keep_prob:0.5})
        if j%100==0:
            #train_accuracy = accuracy1.eval(feed_dict={x1:train_x, y1:train_y, keep_prob:1.0})
            train_accuracy = accuracy1.eval(feed_dict={x1:batch_abs, y1:batch_y, keep_prob:1.0})
            print('step', j, 'accuracy : ', train_accuracy)
            #print('test accuracy : ', sess.run(accuracy1, feed_dict={x2:test_x, y1:test_y, keep_prob:1.0}))
            #print('test accuracy : ', sess.run(accuracy1, feed_dict={x2:batch_imag, y2:batch_y, keep_prob:1.0}))
