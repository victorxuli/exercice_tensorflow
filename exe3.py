# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:26:41 2019

@author: victo
"""

import tensorflow as tf

w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])

y = tf.matmul(w,x)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

saver =tf.train.Saver()

save_path = saver.save(sess,"test")
#
print("save path is :",save_path)


