# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 11:36:50 2019

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


###############################权重项初始化  骨架
n_input = 784
n_output = 10

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_output])
_keepradio = tf.placeholder(tf.float32)



weights = {
        'WC1': tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
        'WC2': tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
        'WD1': tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
        'WD2': tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
}

biais = {
        'bc1': tf.Variable(tf.random_normal([64],stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128],stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024],stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output],stddev=0.1))        
}
##############################前向传播
def conv_basic(_input,_w,_b,_keepradio):
    _input_r = tf.reshape(_input,shape=[-1,28,28,1])
    ##第一层卷积层
    _conv1 = tf.nn.conv2d(_input_r,_w['WC1'], strides = [1,1,1,1],padding='SAME') #strides = 4维格式 batch_size,h,w,channel
    
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))    #激活层
    
    _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides =[1,2,2,1],padding='SAME') #pooling层
     
    _pool_dr1 = tf.nn.dropout(_pool1,_keepradio) ##防止过拟合
#    Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，
#    让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），
#    因为下次样本输入时它可能又得工作了

    ##第二层卷积层
    _conv2 = tf.nn.conv2d(_pool_dr1,_w['WC2'], strides = [1,1,1,1],padding='SAME') #strides = 4维格式 batch_size,h,w,channel
    
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    
    _pool2 = tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides =[1,2,2,1],padding='SAME')
    
    _pool_dr2 = tf.nn.dropout(_pool2,_keepradio) ##防止过拟合
    
    _dense1 = tf.reshape(_pool_dr2,[-1,_w['WD1'].get_shape().as_list()[0]])
    
    ##全连接层
    _fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(_dense1,_w['WD1']),_b['bd1']))
    
    _fc_dr1 = tf.nn.dropout(_fc1,_keepradio)
    
    _out = tf.add(tf.matmul(_fc_dr1,_w['WD2']),_b['bd2'])
    
    out = {'input_r':_input_r, 'conv1':_conv1, 'pool1':_pool1, 'pool_dr1':_pool_dr1, 
           'conv2':_conv2, 'pool2':_pool2, 'pool_dr2':_pool_dr2,'dense1':_dense1, 'fc1':_fc1, 
           'fc1_dr1':_fc_dr1,'out':_out
           }
    
    return out

###########################################反向传播
    
pred = conv_basic(x,weights,biais,_keepradio)['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))  ###交叉熵计算损失函数

optm = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)##梯度下降求最优参数

corr = tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))

accr = tf.reduce_mean(tf.cast(corr,tf.float32))

################################################
init = tf.global_variables_initializer()
print("Function Ready")

training_epochs = 15
batch_size = 100
display_step = 1

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
        total_batch = 20
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    #        feeds = {x:batch_xs,y:batch_ys,_keepradio:0.7}
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,_keepradio:0.7})
            avg_cost = avg_cost+sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,_keepradio:1.})
        avg_cost = avg_cost / total_batch
        feeds_train = {x:batch_xs, y:batch_ys,_keepradio:1.}
#        feeds_test = {x: mnist.test.images, y:mnist.test.labels}
        train_acc = sess.run(accr,feed_dict=feeds_train)
#        test_acc = sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,_keepradio:1.})
        print(" Epoch: %03d/%03d cost: %.9f test_acc: %.3f" 
              %(epoch,training_epochs,avg_cost,train_acc))
        
        if(epoch%save_step==0):
            saver.save(sess,"save/model_cnn.ckpt-"+str(epoch))
###############################################读取模型测试   
do_train=0
if (do_train==0):
    epoch = training_epochs-1
    saver.restore(sess,"save/model_cnn.ckpt-"+str(epoch))
    test_acc = sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,_keepradio:1.})  
    print("accuracy : %.3f" %test_acc)
    






######################################保存模型
#print(sess.run(weights))
#print(sess.run(biais))
#saver = tf.train.Saver()
#saver_path = saver.save(sess,"save/model_cnn.ckpt")
#
#print("Modele saved: ",saver_path)
#
#
######################################模型提取
#saver = tf.train.Saver()
#saver.restore(sess,"save/model_cnn.ckpt")

########

 

