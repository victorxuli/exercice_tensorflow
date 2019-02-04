# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:31:07 2019

@author: victo
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import skimage
from skimage import io as skio
import skimage.color as color


def convert_gray(f):
    rgb=skio.imread(f)
    return color.rgb2gray(rgb)



mnist = input_data.read_data_sets("data/", one_hot = True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

#print(trainimg.shape)
#print(trainlabel.shape)
#print(testimg.shape)
#print(testlabel.shape)
#
#
#print(trainlabel[0])
plt.imshow(trainimg[0,:].reshape(28,28),cmap='gray')

################################################################################

x = tf.placeholder("float",[None,784]) #训练集占位
y = tf.placeholder("float",[None,10])  #样本数占位

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

###########创造逻辑回归模型
actv = tf.nn.softmax(tf.matmul(x,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))

learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##################################################################################
##0:列最大值索引 1：行最大值索引
##预测
pred = tf.equal(tf.argmax(actv,1),tf.arg_max(y,1))
#精度
accr = tf.reduce_mean(tf.cast(pred,"float"))

training_epochs = 150 ##迭代次数
batch_size = 100 ##每进行一次迭代需要的样本数
display_step = 1
###################
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x: batch_xs, y:batch_ys})
        feeds = {x: batch_xs, y:batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
    
    if(epoch%display_step ==0):
        feeds_train = {x:batch_xs, y:batch_ys}
        feeds_test = {x: mnist.test.images, y:mnist.test.labels}
        train_acc = sess.run(accr,feed_dict=feeds_train)
        test_acc = sess.run(accr,feed_dict=feeds_test)
        print(" Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" 
              %(epoch,training_epochs,avg_cost,train_acc,test_acc))
        
print("DONE")

####################加载测试集####################################################



