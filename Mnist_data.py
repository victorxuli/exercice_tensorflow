# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:14:22 2019

@author: victo
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

print("packs loaded")



print("Download and Extract Mnist dataset")

mnist = input_data.read_data_sets("data/", one_hot = True)

print(" type of mnist is %s" %(type(mnist)))

print(" number of train data is %d" %(mnist.train.num_examples))

print(" number of test data is %d" %(mnist.test.num_examples))


trainimg = mnist.train.images

trainlabel = mnist.train.labels

testimg = mnist.test.images

testlabel = mnist.test.labels




