from __future__ import absloute_import 
from __future__ import division
from __future__ import print_function

import argparse 
import sys

import tensorflow as tf 
import cv2 
import numpy as np 
import numpy 

im[]
im_linear[]
y_label[]
y_linear[]
num_images = 
factor =
learning_rate = 					#with face
num_pos_images =                    #for the regression
filename1 = 'annotation1.csv'      #with face
filename2 = 'annotation2.csv'	   #without face but I think there is no use of annotation 2 as we dont require it for labeling as well regression
raw_data1 = open (filename1, 'rt')
raw_data2 = open (filename2, 'rt')
data1 = numpy.loadtxt(raw_data1, delimiter = ",")
data2 = numpy.loadtxt(raw_data2, delimiter = ",")

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')   


for i in range(0,num_images):
	if(i%factor == 0):	#depending on the number of images with the label we will change this factor of 2
		name = ''  #depending on the folder of images with no human
		img = cv2.imread(name)
		img = cv2.resize(img,(100,100))
		im.append(img)
		y_label.append(0)
	else : 
		name  = ''	#depends on the folder
		img = cv2.imread(name)
		img = cv2.resize(img,(100,100))
		a = [data1[(i/factor) + (i % factor)][0],data1[(i/factor) + (i % factor)][1],data1[(i/factor) + (i % factor)][2],data1[(i/factor) + (i % factor)][3]]
		im.append(img)
		y_label.append(1)   #change the value of factor(2) if the factor is different 
		y_linear.append(a)
		im_linear.append(img)

X_in = np.array(im)
Y_in = np.array(y_label)
Y_in_linear = np.array(y_linear)
im_in_linear = np.array(im_linear)
print X_in.shape
im_in_linear = np.reshape(im_in_linear, (num_pos_images, X_in.shape[1]*X_in.shape[2]*X_in.shape[3]))
X_in = np.reshape(X_in, (num,images, X_in.shape[1]*X_in.shape[2]*X_in.shape[3]))
Y_in = np.reshape(Y_in, (num_images, 1))
Y_in_linear = np.reshape(Y_in_linear,(num_pos_images, 4))
print X_in.shape




W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])

X = tf.placeholder(tf.float32,[NONE, X_in.shape[1]])
Y = tf.placeholder(tf.float32. [NONE, 1])

X_linear = tf.placeholder(tf.float32, [NONE, im_in_linear.shape[1]])
Y_linear1 = tf.placeholder(tf.float32, [NONE, 4])	

X_image = tf.reshape(X, [-1,100,100,3])

h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1)+b_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2)+b_conv2)
h_poll1 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_poll1, W_conv3)+b_conv3)
h_poll2 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5,5,128,256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_poll2, W_conv4)+b_conv4)
h_poll3 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([25*25*256, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_poll3,[-1, 25*25*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)

W_fc2_linear = weight_variable([1024,4])
b_fc2_linear = bias_variable([4])

W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2]) 

y_conv = tf.matmul(h_fc1, W_fc2)+b_fc2
y_conv_linear = tf.matmul(h_fc1, W_fc2_linear)+b_fc2_linear 

cost = tf.reduce_sum(tf.pow(y_conv_linear-Y_in_linear)/(2*num_pos_images)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
predicted=Y if ((y_conv>0.5 and Y==1) or (y_conv<0.5 and Y==0)) else (1-Y)
correct_prediction=tf.equal(predicted,Y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100
with tf.Session() as sess:
	sess.run(tf.gloabal_variables_initializer())
	saver = tf.train.saver()
	#*******Training******
	for i in range(20000):   # change according to the number of images
		train_step.run(feed_dict{X:X_in, Y:Y_in})
		sess.run(optimizer,feed_dict= {X_linear: im_in_linear,Y_linear1: Y_in_linear })


