#tensorflow v1.15

import os
import sys
import csv
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

starting_index = 0

def one_hot(label, num_label):
    result = [0] * num_label
    result[int(label)] = 1
    return result

def weight_variable_fc(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def weight_variable(filter_size, in_channels, out_channels):
    shape = filter_size[:]
    shape.extend([in_channels, out_channels])
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')     #input shape = [batch, in_height, in_width, in_channels], filter shape = [filter_height, filter_width, in_channels, out_channels]

def max_pool(in_shape, filter_h, filter_w):
    return tf.nn.max_pool(in_shape, ksize=[1, filter_h, filter_w, 1], strides=[1, filter_h, filter_w, 1], padding='SAME')

def read_data(directory):
    sub_dir_list = os.listdir(directory)
    label = 0
    result = []
    for sub_dir in sub_dir_list:
        num_class = len(sub_dir_list)
        cur_dir = directory + sub_dir +'/'
        file_list = os.listdir(cur_dir)
        print(sub_dir.rjust(15),':', len(file_list), 'files')
        for cur_file in file_list:
            file_name = cur_dir + cur_file
            with open(file_name) as f:
                csv_reader = csv.reader(f)
                data = []
                tmp = []
                for i in csv_reader:
                    data.append(i[1])
                tmp.append(data)
                tmp.append(label)
            result.append(tmp)
        label += 1
    print('number of class :', num_class, 'classes')
    print('number of data in one csv file : ', len(result[0][0]))
    np.random.shuffle(result)
    return result, num_class

def array_to_list(array_data):
    result = []
    for dataset in array_data:
        data = dataset[0]
        label = dataset[1]
        data.append(label)
        result.append(data)
    return result

def get_batch(batch, data_x, data_y):
    global starting_index 
    batch_x = []
    batch_y = []
    if starting_index + batch < len(data_x):
        batch_x = data_x[starting_index : starting_index + batch]
        batch_y = data_y[starting_index : starting_index + batch]
        starting_index += batch
    else:
        rest = starting_index + batch - len(data_x)
        batch_x = data_x[starting_index : len(data_x)]
        batch_y = data_y[starting_index : len(data_y)]
        np.append(data_x, data_x[0:rest])
        np.append(data_y, data_y[0:rest])
        starting_index = rest
    return batch_x, batch_y

