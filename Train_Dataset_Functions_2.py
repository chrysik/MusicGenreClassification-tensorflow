import os, glob
import numpy as np
import re, pickle, cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize

def create_track_paths(Genre_name,file_types,current_file_path,dataset_path):
	data_path=[]
	for i in Genre_name:
		data_path.append(os.path.abspath(current_file_path+dataset_path+i))
	print("Data path:",data_path)

	genre_paths=[{} for i in range(len(Genre_name))]
	n=0
	path=[]
	all_genre_paths=[]
	for i in genre_paths:
		i['genre']=Genre_name[n]
		i['genre_path']=data_path[n]
		for k in file_types:
			path.append(sorted(glob.glob(i['genre_path']+k)))
		for j in path:
			all_genre_paths.append(j)
			i['all_genre_paths']=all_genre_paths
		n+=1
		path=[]
		all_genre_paths=[]

	conc_paths=[]
	for i in genre_paths:
		conc_paths=i['all_genre_paths'][0]+i['all_genre_paths'][1]
		i['all_genre_conc_paths']=conc_paths



	return genre_paths

def create_labels(genre_paths):
	chunk_path=[]
	labels=[]
	for i in genre_paths:
		n=0
	#-Create new directory to save chunk-s#
		chunk_path=os.path.join(i['genre_path'],"chunk")
		i['chunk_paths']=(sorted(glob.glob(chunk_path+"/*.flac")))
		label_genre=i['genre']
		for w in range(len([name for name in os.listdir(chunk_path)])):
			labels.append(label_genre)
	with open("labels_500.pkl","wb") as f:
		pickle.dump(labels,f)
	print("Number of labels",len(labels))
	labels=np.array(labels)

	return labels



def input_to_clasif(openfeaturepickle):
	input_to_clas=[]
	with (open(openfeaturepickle,"rb")) as open_feature_pickle:
		while True:
			try:
				input_to_clas.append(pickle.load(open_feature_pickle))
			except EOFError:
				break
	input_to_clas=np.array(input_to_clas)

	return input_to_clas



def output_to_clas(labels):
	#########--One hot encode labels--#########
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(labels)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

	output_to_clas=onehot_encoded

	return output_to_clas