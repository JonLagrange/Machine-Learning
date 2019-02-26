# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# %matplotlib inline  # 在使用jupyter notebook 或者 jupyter qtconsole的时候，才会经常用到%matplotlib
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    # 读入压缩包里第一个文件的所有内容，并以空格分割，形成一个很大的list
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print('Data size %d' % len(words))


vocabulary_size = 50000  # 只统计50000-1个常用词，剩下的词统称UNK，即Unknow的缩写

def build_dataset(words):
  count = [['UNK', -1]]  # count就是包括UNK在内的所有50000-1个常用词的词语和出现次数
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # 按照count里的词先后顺序，给词进行编号，UNK是0，出现最多的the是1，出现第二多的of是2
  dictionary = dict()  # dictionary就是词到编号的对应关系
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]  # 可以快速查找词的编号：index = dictionary(word)
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)  # data是把原文的词都转化成对应编码以后的串
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # reverse_dictionary则是编号到词的对应关系，可以快速查找某个编号是什么词
  return data, count, dictionary, reverse_dictionary  # 保存dictionary和reverse_dictionary这一点十分值的学习，对于频繁的查询，这样的缓存能大大增加速度。如果用函数的方式，每次查都要轮询，就土了

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


data_index = 0  # 使用全局变量data_index来记录当前取到哪了，每次取一个batch后会向后移动，如果超出结尾，则又从头开始

def generate_batch(batch_size, num_skips, skip_window):
  global data_index  # #global关键字(内部作用域想要对外部作用域的变量进行修改)
  assert batch_size % num_skips == 0  # num_skips代表着我们从skip_window中选取多少个不同的词作为我们的（input， output）
  assert num_skips <= 2 * skip_window  # skip_window是确定取一个词周边多远的词来训练，比如说skip_window是2，则取这个词的左右各两个词，来作为它的上下文词。后面正式使用的时候取值是1，也就是只看左右各一个词

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):  # 运算符//取整除，返回商的整数部分（向下取整）
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels



print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.  嵌入向量的维数
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# 我们选择一个随机验证集来对最近邻居进行抽样。我们将验证样本限制为具有较低数字ID的单词，这也是最常用的
valid_size = 16  # Random set of words to evaluate similarity on.  用于评估相似性的随机词集
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()  # 创建了一个新的图graph，然后把graph设为默认，那么接下来的操作不是在默认的图中，而是在graph中了。你也可以认为现在graph这个图就是新的默认的图了

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  # valid_dataset是用来人工验证的小数据集，是constant，直接赋值前面生成的valid_examples

    # Variables.
    embeddings = tf.Variable(  # embeddings是用来存储Embeddings向量空间的变量，初始化成-1到1之间的随机数，后面优化时调整。这里它是一个 50000 * 128 的二维变量
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  # random_uniform从正态分布中输出随机值
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],  # truncated_normal从截断的正态分布中输出随机值。生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))  # softmax_weights 和 softmax_biases 是用来做线性逻辑分类的参数

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)  # 通过tf.nn.embedding_lookup可以直接根据embeddings表(50000,128)，取出一个与输入词对应的128个值的embed，也就是128维向量。其实是一batch同时处理，但说一个好理解一些
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)  # 优化器，这里使用了AdagradOptimizer，当然也可以使用其他SGD、Adam等各种优化算法，Tensorflow都实现同样的接口，只需要换个函数名就可以

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))  # reduce_sum应该理解为压缩求和，用于降维，参数1表示按行求和
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))  # 计算一个valid_dataset中单词的相似度


num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()


num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
  pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)