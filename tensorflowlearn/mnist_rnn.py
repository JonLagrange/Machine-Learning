import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.contrib import  rnn

mnist=input_data.read_data_sets("./data",one_hot=True)

train_rate=0.001
train_step=10001
batch_size=128
display_step=100

frame_size=28
sequence_length=28
hidden_num=100
n_classes=10
layer_num=2

#定义输入,输出
x=tf.placeholder(dtype=tf.float32,shape=[None,sequence_length*frame_size],name="inputx")
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name="expected_y")
#定义权值
weights=tf.Variable(tf.truncated_normal(shape=[hidden_num,n_classes]))
bias=tf.Variable(tf.zeros(shape=[n_classes]))

def LSTM(x,weights,bias):
    x=tf.reshape(x,shape=[-1,sequence_length,frame_size])
    #rnn_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_num, state_is_tuple=True)
    #rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell for _ in range(layer_num)], state_is_tuple=True)
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_num) for _ in range(layer_num)])
    #init_state=tf.zeros(shape=[batch_size,rnn_cell.state_size])
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    # 其实这是一个深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元
    output,states=tf.nn.dynamic_rnn(rnn_cell,x,initial_state=init_state,dtype=tf.float32)
    logit=tf.matmul(output[:,-1,:],weights)+bias
    predy=tf.nn.softmax(logit)
    return logit, predy

_logit, _predy=LSTM(x,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_logit,labels=y))
train=tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(_predy,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.to_float(correct_pred))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
testx,testy=mnist.test.next_batch(batch_size)

for step in range(train_step):
    batch_x,batch_y=mnist.train.next_batch(batch_size)
#    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})

    if step % display_step ==0:
        acc,loss=sess.run([accuracy,cost],feed_dict={x:testx,y:testy})
        print(step,acc,loss)