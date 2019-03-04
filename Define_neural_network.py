import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import os
from nn_config import test_size,n_epochs,batch_size,learning_rate,filter_shape_1,filter_shape_2,filter_shape_3,pool_shape,num_channels,fc_neurons_size,num_filters1,num_filters2,num_filters3


def convolution_layer(input_data,num_input_channels,filter_shape,num_filters,pool_shape):
	convol_filter_shape = [filter_shape[0],filter_shape[1],num_input_channels,num_filters]
	weights = tf.Variable(tf.truncated_normal(convol_filter_shape),name='Weights_conv')
	bias = tf.Variable(tf.truncated_normal([num_filters]),name='Bias_conv')
	out_conv_layer = tf.nn.conv2d(input=input_data, filter=weights,strides=[1,1,1,1],padding='SAME',name='convolution')
	out_conv_layer += bias
	#relu non-linear activation
	out_conv_layer = tf.nn.relu(out_conv_layer,name='relu')
	#max pooling
	ksize = [1,pool_shape[0],pool_shape[1],1]
	strides = [1,2,2,1]
	out_conv_layer = tf.nn.max_pool(value=out_conv_layer,ksize=ksize,strides=strides,padding='SAME',name='max_pooling')
	return out_conv_layer,weights,bias

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flattened = tf.reshape(layer,[-1,num_features])
	return layer_flattened, num_features

def fully_con_layer(input_to_fc,num_inputs,num_outputs):
	weight_fc = tf.Variable(tf.truncated_normal([num_inputs,num_outputs]),name='Weights_fc')
	bias_fc = tf.Variable(tf.truncated_normal([num_outputs]),name='Bias_fc')
	dense_layer = tf.add(tf.matmul(input_to_fc,weight_fc),bias_fc,name='dense')
	return dense_layer
	
def def_nn(Y,X_shaped,n_classes,filter_shape_1,filter_shape_2,filter_shape_3,num_filters1,num_filters2,num_filters3,pool_shape):
	with tf.device('/cpu:0'):
		layer_1,weights_conv_1,bias_1 = convolution_layer(input_data=X_shaped,num_input_channels=1,filter_shape=filter_shape_1,num_filters=num_filters1,pool_shape=pool_shape)
		layer_2,weights_conv_2,bias_2 = convolution_layer(input_data=layer_1,num_input_channels=num_filters1,filter_shape=filter_shape_2,num_filters=num_filters2,pool_shape=pool_shape)
		layer_3,weights_conv_3,bias_3 = convolution_layer(input_data=layer_2,num_input_channels=num_filters2,filter_shape=filter_shape_3,num_filters=num_filters3,pool_shape=pool_shape)

		layer_flattened,num_features = flatten_layer(layer_3)

		dense_layer_1 = fully_con_layer(input_to_fc=layer_flattened,num_inputs=num_features,num_outputs=fc_neurons_size)
		dense_layer_1 = tf.nn.relu(dense_layer_1,name='dense1_relu')

		dense_layer_2 = fully_con_layer(input_to_fc=dense_layer_1,num_inputs=fc_neurons_size,num_outputs=n_classes)
		pred  =  tf.argmax(dense_layer_2, 1, name="pred")
		Y_predicted  =  tf.nn.softmax(dense_layer_2,name='prediction')
		
		cost  =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer_2, labels=Y))
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	return Y_predicted,cost,optimizer

