import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from Configuration import config
from datetime import datetime
from nn_config import test_size,n_epochs,batch_size,learning_rate,filter_shape_1,filter_shape_2,filter_shape_3,pool_shape,num_channels,fc_neurons_size,num_filters1,num_filters2,num_filters3
from Define_neural_network import convolution_layer,flatten_layer,fully_con_layer,def_nn
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from Train_Dataset_Functions_2 import create_track_paths,create_labels,input_to_clasif,output_to_clas
import pickle 
import ast
import os


def mkdir_to_save_model():
	now = datetime.now()
	print(now)
	model_path = current_file_path+"\Model_"+ str(now.day)+'-'+str(now.month)+'-'+str(now.year) + "_"+ str(now.hour)+"-"+str(now.minute)+'/'
	try:
		os.makedirs(model_path)
	except OSError as e:
		print("Error in creating directory")
	return model_path

Hz, chunk_len_sec, Width, Height, file_types, Genre_name, current_file_path  =  config()
genre_paths = []
labels = []
with (open("labels_500.pkl","rb")) as open_pickle:
	while True:
		try:
			labels.append(pickle.load(open_pickle))
		except EOFError:
			break
labels = labels[0]
input_to_clas = input_to_clasif("dataset_CEHPR_500_mfcc.pkl")
output_to_clas = output_to_clas(labels)
print(type(output_to_clas))

nn_input = input_to_clas[0]
nn_output = output_to_clas
n_classes = len(set(labels))
image_flat_size = Width*Height
image_shape = (Width,Height)
input_size = nn_input[0].size

model_path = mkdir_to_save_model()
print(np.shape(nn_input))
nn_input_new = nn_input[:,:,:,np.newaxis]
nn_input = nn_input_new
print(np.shape(nn_input))

x_train,x_test,y_train,y_test  =  train_test_split(nn_input,nn_output,test_size=test_size)
X = tf.placeholder(tf.float32,[None,Width,Height,1],name='X')
Y = tf.placeholder(tf.float32,[None,n_classes],name='Y_real')

Y_predicted,cost,optimizer = def_nn(Y,X,n_classes,filter_shape_1,filter_shape_2,filter_shape_3,num_filters1,num_filters2,num_filters3,pool_shape)

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(n_epochs):
		epoch_loss = 0
		i = 0
		while i<len(x_train):
			start = i
			end = i+batch_size
			batch_x = np.array(x_train[start:end])
			batch_y = np.array(y_train[start:end])
			_,c = sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
			epoch_loss += c
			i += batch_size
		print("Epoch",epoch+1,"completed out of", n_epochs,"loss:",epoch_loss)
	correct_results = tf.equal(tf.argmax(Y_predicted,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_results,tf.float32))
	ac_train = sess.run(accuracy, feed_dict={X:x_test,Y:y_test})
	print("Accuracy for test images:",ac_train)
	print("\n--------- Training complete ---------")
	saver.save(sess, model_path + " model")
