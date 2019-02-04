# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:17:07 2019

@author: victo
"""

import tensorflow as tf



################ tessorflow加法 ########################

state = tf.Variable(0)

new_value = tf.add(state,tf.constant(1))

update = tf.assign(state,new_value)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


print(sess.run(state))

for i in range(3):
    sess.run(update)
    print(sess.run(state))