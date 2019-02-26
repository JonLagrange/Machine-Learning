import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class hello_world:
    def __init__(self):
        self.mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

    def inference(self):
        self.input = tf.placeholder(tf.float32, [None, 784])         #输入28*28的图
        self.label = tf.placeholder(tf.float32, [None, 10])          #正确的分类标签
        with tf.variable_scope('model'):
            w = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))            #权重参数
            b = tf.Variable(tf.zeros([10]))         #偏置参数
            y = tf.nn.softmax(tf.matmul(self.input, w) + b)          # 10个分类输出（0-9数字）

        with tf.variable_scope('optimize'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(y), reduction_indices=[1]))       #使用交叉熵获得损失函数
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)         #使用梯度下降法最小化损失函数

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.label, 1))        #预测值是否正确
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))         #求正确率

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):               #训练1000轮，每次100条数据
                batch_data, batch_label = self.mnist.train.next_batch(100)
                sess.run(self.train_step,{self.input:batch_data, self.label:batch_label})
            print('accuracy : %f' % sess.run(self.accuracy,{self.input: self.mnist.test.images, self.label: self.mnist.test.labels}))


if __name__ == '__main__':
    power = hello_world()
    power.inference()
    power.train()