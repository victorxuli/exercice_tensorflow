# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:27:56 2019

@author: victo
"""

import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

##############################加载样本数据集
mnist = input_data.read_data_sets("data/", one_hot = True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

##############################参数初始化
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

#########################################网络结构
stddev=0.1
weights = {
        'W1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
        'W2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}

biais = {
        'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }
#########################################前向传播
def multilayer_perception(_X,_weights,_biais):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['W1']),_biais['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['W2']),_biais['b2']))
    return (tf.matmul(layer_2,_weights['out']+biais['out']))

###########################################反向传播
pred = multilayer_perception(x,weights,biais)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))  ###交叉熵函数

optm = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)##梯度下降求最优参数

corr = tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))

accr = tf.reduce_mean(tf.cast(corr,"float"))

##############################################
init = tf.global_variables_initializer()
print("Function Ready")

training_epochs = 60
batch_size = 100
display_step = 1

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x:batch_xs,y:batch_ys}
        sess.run(optm,feed_dict=feeds)
        avg_cost = avg_cost+sess.run(cost,feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    feeds_train = {x:batch_xs, y:batch_ys}
    feeds_test = {x: mnist.test.images, y:mnist.test.labels}
    train_acc = sess.run(accr,feed_dict=feeds_train)
    test_acc = sess.run(accr,feed_dict=feeds_test)
    print(" Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" 
          %(epoch,training_epochs,avg_cost,train_acc,test_acc))
print("DONE")
        
