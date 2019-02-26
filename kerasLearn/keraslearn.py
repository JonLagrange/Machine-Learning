from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2, 6, 200)  #在默认情况下，linspace函数可以生成元素为50的等间隔数列。而前两个参数分别是数列的开头与结尾。如果写入第三个参数，可以制定数列的元素个数。
np.random.shuffle(X)  #numpy.random,shuffle(x)是进行原地洗牌，直接打乱顺序，改变x的值，而无返回值
Y = 0.5 * X + 2 + 0.15 * np.random.randn(200, )

# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # train first 160 data points
X_test, Y_test = X[160:], Y[160:]  # test remaining 40 data points

model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))

model.compile(loss='mse', optimizer='sgd')  #损失函数=均方误差‘mse’，优化器=随机梯度下降'sgd'
model.fit(X_train, Y_train, epochs=100, batch_size=64)

print('\nTesting ------------')
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', loss_and_metrics)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()