import tensorflow as tf

x = tf.ones((1, 3))
y = tf.ones((1, 3))
y1 = tf.layers.dense(x, 4, name='h1')
y2 = tf.layers.dense(y, 4, name='h1', reuse=True)

# y1 and y2 will evaluate to the same values
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y1))
print(sess.run(y2))  # both prints will return the same values