# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:42:34 2019

@author: victo
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


####随机生成1000个点 围绕在 y = 0.1x + 0.3
num_points = 1000

vecteur_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1*0.1 +0.3+np.random.normal(0.0,0.03)
    vecteur_set.append([x1,y1])



### 生成样本
    
x_data = [v[0] for v in vecteur_set]
y_data = [v[1] for v in vecteur_set]

plt.scatter(x_data,y_data,c='r')
plt.title('distribution of samples')

####构建线性回归模型############################################################

W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name = 'W')

b = tf.Variable(tf.zeros([1]),name='b')

y = W*x_data+b

#### 计算均方误差 loss_function ###################
loss = tf.reduce_mean(tf.square(y-y_data),name='loss')
#####梯度下降优化误差  学习率:0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
#####最小化误差
train = optimizer.minimize(loss,name='train')

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print("W initial = ",sess.run(W),  "b initial = ",sess.run(b), " loss = ",sess.run(loss))

for i in range (50):
    sess.run(train)
    print("step : ",i,"W = :",sess.run(W),  "b = :",sess.run(b), "loss = ",sess.run(loss))
    
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))



