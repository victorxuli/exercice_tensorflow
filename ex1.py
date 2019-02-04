# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:02:59 2019

@author: victo
"""

import tensorflow as tf

###创建变量矩阵
a = tf.Variable([[0.5,1.0]])

b = tf.Variable([[1.5],[2.0]])

## 矩阵相乘
y = tf.matmul(a,b)

init_op = tf.global_variables_initializer()

##########################################################################
sess = tf.Session()
sess.run(init_op)
##########################################################################
print(sess.run(y))

c = tf.zeros([2,3])

print(sess.run(c))

d = tf.zeros_like(a)

print(sess.run(d))

e = tf.constant([1,2,3,4,5,6])

print(sess.run(e))

f = tf.range(1,10,1)

print(sess.run(f))

norm = tf.random_normal([2,3],mean=1,stddev=4)

print(sess.run(norm))

shuff = tf.random_shuffle(e)

print(sess.run(shuff))



