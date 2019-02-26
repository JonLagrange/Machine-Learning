#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 08:32:59 2018
@author: yhilly
"""

import tensorflow as tf

labels = [[0.2, 0.3, 0.5],
          [0.1, 0.6, 0.3]]
logits = [[4, 1, -2],
          [0.1, 1, 3]]

logits_scaled = tf.nn.softmax(logits)
# 注意！！！这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,就是对向量里面所有元素求和，最后才得到，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！
result = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    print(sess.run(logits_scaled))
    print(sess.run(result))
