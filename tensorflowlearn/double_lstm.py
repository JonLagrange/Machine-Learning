# -*-coding:utf-8-*-
import numpy as np
import tensorflow as tf

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIME_STEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    X = []  # ninihaoniaho zh
    Y = []

    for i in range(len(seq) - TIME_STEPS):
        X.append([seq[i:i + TIME_STEPS]])
        Y.append([seq[i + TIME_STEPS]])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def lstm_model(X, Y, is_training):
    # cell_unit = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    output = outputs[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)
    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=Y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adagrad', learning_rate=0.1)

    return predictions, loss, train_op


def train(sess, train_X, train_Y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, Y = ds.make_one_shot_iterator().get_next()

    ### use model get result
    with tf.variable_scope('model'):
        predictions, loss, train_op = lstm_model(X, Y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print('train step:' + str(i) + ',loss:' + str(l))


def run_eval(sess, test_X, test_y):
    # print('------------runned')
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIME_STEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIME_STEPS) * SAMPLE_GAP

train_X, train_Y = generate_data(np.sin(np.linspace(
    test_start, test_end, TRAINING_EXAMPLES + TIME_STEPS, dtype=np.float32)))
test_X, test_Y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIME_STEPS, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_Y)
    run_eval(sess, test_X, test_Y)
