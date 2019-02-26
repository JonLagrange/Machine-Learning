import numpy as np
import tensorflow as tf

b = tf.constant(np.random.normal(size=(3, 4)))

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(tf.multinomial(b, 1)))
