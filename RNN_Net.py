# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:02:26 2019

@author: victo
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

##############################加载样本数据集
mnist = input_data.read_data_sets("data/", one_hot = True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

diminput = 28
dimhidden = 128
dimoutput = 10
nsteps = 28

training_epochs = 15
batch_size = 16
display_step = 1

################################权重项初始化  骨架
weight = {
        'hidden': tf.Variable(tf.random_normal([diminput,dimhidden])),
        'out': tf.Variable(tf.random_normal([dimhidden,dimoutput]))
}

biaises = {
        'hidden': tf.Variable(tf.random_normal([dimhidden])),
        'out': tf.Variable(tf.random_normal([dimoutput]))        
}

##################################### RNN模型

def _RNN(_X,_W,_b,_nsteps,_name):
    _X = tf.transpose(_X,perm=[1,0,2])   ##
    
    _X = tf.reshape(_X,[-1,diminput])
#    _X = tf.reshape(_X,[batch_size*_nsteps,diminput])

    _H = tf.add(tf.matmul(_X,_W['hidden']),_b['hidden'])

    _Hsplit = tf.split(_H,_nsteps,0)

#    with tf.variable_scope('forward'):
#        lstm_fw_cell = rnn_cell.BasicLSTMCell(dim_hidden)   
#    with tf.variable_scope('backward'):
#        lstm_bw_cell = rnn_cell.BasicLSTMCell(dim_hidden)

    with tf.variable_scope(_name) as scope:
#    scope = tf.variable_scope(_name)
        scope.reuse_variables()
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden)
        _LSTM_O, _LSTM_S = tf.nn.static_rnn(lstm_cell,_Hsplit,dtype=tf.float32)
        
    _O = tf.matmul(_LSTM_O[-1],_W['out']) +_b['out']
    
    return {
            'X':_X, 'H':_H, 'Hsplit':_Hsplit, 'LSTM_S':_LSTM_S,
            'LSTM_O':_LSTM_O, 'O':_O
            }
print('Network ready')
        
###########################################   
n_input = 784
n_output = 10
x = tf.placeholder("float",[None,nsteps,diminput])
y = tf.placeholder("float",[None,n_output])

myrnn = _RNN(x,weight,biaises,nsteps,'basic')

pred = myrnn['O']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))  ###交叉熵计算损失函数

optm = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)##梯度下降求最优参数

#corr = tf.equal(tf.argmax(pred,1),tf.arg_max(y,1), tf.float32)

accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.arg_max(y,1)),tf.float32))

init = tf.global_variables_initializer()
print("Function Ready")

##########################################################

save_step = training_epochs-1 ##保存间隔 隔1个epoch保存
saver = tf.train.Saver(max_to_keep=1)
do_train = 1

sess = tf.Session()
sess.run(init)
###########################################迭代训练

if(do_train==1):
    for epoch in range(training_epochs):
        avg_cost = 0
    #    total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = 100
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(batch_size,nsteps,diminput)
    #        feeds = {x:batch_xs,y:batch_ys,_keepradio:0.7}
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
            avg_cost = avg_cost+sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})
        avg_cost = avg_cost / total_batch
        
        feeds_train = {x:batch_xs, y:batch_ys}
        
#        feeds_test = {x: mnist.test.images, y:mnist.test.labels}
        train_acc = sess.run(accr,feed_dict=feeds_train)
#        test_acc = sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,_keepradio:1.})
        print(" Epoch: %03d/%03d cost: %.9f train_acc: %.3f" 
              %(epoch,training_epochs,avg_cost,train_acc))
        
        if(epoch%save_step==0):
            saver.save(sess,"save/rnn/model_rnn.ckpt-"+str(epoch))
###############################################读取模型测试   
do_train=0
if (do_train==0):
    epoch = training_epochs-1
    saver.restore(sess,"save/rnn/model_rnn.ckpt-"+str(epoch))
    test_acc = sess.run(accr,feed_dict={x:batch_xs,y:batch_ys})  
    print("accuracy : %.3f" %test_acc)




    
        