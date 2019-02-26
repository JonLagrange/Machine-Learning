#coding=utf-8
import tensorflow as tf
import numpy as np

hello=tf.constant('Hello TensorFlow')
sess=tf.Session()
print(sess.run(hello))