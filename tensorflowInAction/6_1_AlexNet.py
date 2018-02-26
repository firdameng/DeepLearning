# coding=utf-8
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batchs = 100


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32,
                                                 stddev=1e-1,
                                                 name='weights'))
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        '''将这一层可训练的参数添加到parameters中'''
        parameters += [kernel, biases]
        # lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        # 注意这里的valid池化模式
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
        print_activations(pool1)

    '''改动了卷积核数，和步长'''
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32,
                                                 stddev=1e-1,
                                                 name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]

        # lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        # 注意这里的valid池化模式
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1,
                                                 name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    '''注意这里没有lrn,和池化'''

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1,
                                                 name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1,
                                                 name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters


def time_tensorflow_run(session, target, info_string):
    num_step_burn_in = 10
    total_duration = 0.0
    total_duration_squard = 0.0
    for i in range(num_batchs + num_step_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_step_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_step_burn_in, duration))
            total_duration += duration
            total_duration_squard += duration * duration
    '''每轮batch平均耗时'''
    mn = total_duration / num_batchs
    '''这是样本方差吧？'''
    vr = total_duration_squard / num_batchs - mn * mn
    print(vr)
    vr = (total_duration_squard - num_batchs * mn * mn) / (num_batchs - 1)
    print(vr)
    sd = math.sqrt(vr)
    print('%s: %s across %d step, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batchs, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        '''这是在模拟真实图片像素值'''
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, pool5, 'forward')
        '''output = sum(t ** 2) / 2  方便下一步对参数求导'''
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'forward-backward')


run_benchmark()
