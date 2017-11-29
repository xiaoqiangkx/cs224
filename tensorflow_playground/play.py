# -*- encoding: utf-8 -*-

"""
@version: 0.1
@author: Clark.wang
@contact: Clark.wang@liulishuo.com
@file: play.py
@time: 03/11/2017 9:51 AM
"""

import tensorflow as tf
import numpy as np


B = tf.Variable(tf.zeros((100, )))
W = tf.Variable(tf.random_uniform((784, 100), -1, 1))

X = tf.placeholder(tf.float32, (100, 784))

h = tf.nn.relu(tf.matmul(X, W) + B)

prediction = tf.nn.softmax(h)
label = tf.placeholder(tf.float32, [100, 10])

cross_entropy = -tf.reduce_mean(tf.reduce_sum(label * tf.log(prediction), axis=1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in xrange(1000):
    batch_x, batch_label = data.next_batch()
    sess.run(h, {X: batch_x, label: batch_label})
