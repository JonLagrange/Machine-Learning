from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

'''
# 这部分返回一个张量
inputs = Input(shape=(784,))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(data, labels)  # 开始训练
'''

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
x = tf.reshape(x, [1, 2, 3, 1])  # def reshape(tensor, shape, name=None),第1个参数为被调整维度的张量,第2个参数为要调整为的形状,返回一个shape形状的新tensor

valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # tf.nn.max_pool(value, ksize, strides, padding, data_format, name),ksize:池化窗口的大小,strides:窗口在每一个维度上滑动的步长
same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

print(valid_pad.get_shape())
print(same_pad.get_shape())