import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data

# create a tf session，and register with keras。
sess = tf.Session()
K.set_session(sess)

# this place holder is the same with input layer in keras
img = tf.placeholder(tf.float32, shape=(None, 784))
# keras layers can be called on tensorflow tensors
x = Dense(128, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

# label
labels = tf.placeholder(tf.float32, shape=(None, 10))
# loss function 当使用categorical_crossentropy损失函数时，你的标签应为多类模式，例如如果你有10个类别，每一个样本的标签应该是一个10维的向量，该向量在对应有值的索引位置为1其余为0
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    for i in range(1000):
        batch = mnist_data.train.next_batch(50)
        _loss, _ = sess.run([loss,train_step], feed_dict={img: batch[0], labels: batch[1]})
        #train_step.run(feed_dict={img: batch[0], labels: batch[1]})

acc_value = accuracy(labels, preds)
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels}))

