# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:34:53 2019

@author: victo
"""

import tensorflow as tf

import numpy as np

a = np.zeros((3,3))

ta = tf.convert_to_tensor(a)

sess = tf.Session()

print(sess.run(ta))

input1 = tf.placeholder(tf.float32) ### input1 : 占位符

input2 = tf.placeholder(tf.float32) ### input2 : 占位符

output = tf.multiply(input1,input2)

s = sess.run([output],feed_dict={input1:[7.],input2:[5.]})

print(s)

