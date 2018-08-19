import os, glob
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile, librosa, librosa.display
import re, pickle, cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize


def create_chunks(chunk_len_sec,genre_paths):
	chunk_path=[]
	labels=[]
	for i in genre_paths:
		n=0
	#-Create new directory to save chunk-s#
		chunk_path=os.path.join(i['genre_path'],"chunk")
		print('Chunk path:',chunk_path)
		try:
			os.makedirs(chunk_path)
		except OSError as e:
			 print("Error in creating directory")
		os.chdir(chunk_path)
	#-Create chunks-#
		for song in i["all_genre_conc_paths"]:
			myaudio=AudioSegment.from_mp3(song)
			chunks=make_chunks(myaudio,chunk_len_sec)
	#-Export all of the individual chunks as flac files-#
			for l, chunk in enumerate(chunks):
				n+=1
				chunk_name="("+str(n)+")"+i['genre']+".flac"
				print("exporting",chunk_name)
				chunk.export(chunk_name,format="flac")
		
	

def chunks_into_nparrays(current_file_path,genre_paths,pickle_waveform_file,Hz):
	nparray=0
	waveform_data=[]
	for i in genre_paths:
		for j in i['chunk_paths']:
			data,rate=librosa.load(j,sr=Hz)
			waveform_data.append(data.astype(np.float32))
			nparray+=1
			print(nparray)
	os.chdir(current_file_path)

	with open(pickle_waveform_file,"wb") as f:
		pickle.dump(waveform_data,f)


def compute_feature(pickletoopen,pickletowrite,Width,Height,feature):
	wf_DATA=[]
	with (open(pickletoopen,"rb")) as open_wfdata_pickle:
		while True:
			try:
				wf_DATA.append(pickle.load(open_wfdata_pickle))
			except EOFError:
				break
	wf_DATA=np.array(wf_DATA)
	print(len(wf_DATA[0]),type(wf_DATA))

	#######--Compute melspectrogram-########
	nn_input=[]
	count=0
	feature_data=[]
	for i in wf_DATA[0]:
		if feature=="spectrogram":
			D = np.abs(librosa.stft(i))
			feature_data.append(D)

		if feature=="log power spectrogram":
			D = np.abs(librosa.stft(i))
			feature_data.append(librosa.power_to_db(D))

		if feature=="log power spectrogram 2":
			D = np.abs(librosa.stft(i))
			feature_data.append(librosa.power_to_db(D**2))

		if feature=="log power spectrogram 4":
			D = np.abs(librosa.stft(i))
			feature_data.append(librosa.power_to_db(D**4))

		if feature=="melspectrogram":
			D=librosa.feature.melspectrogram(i)
			feature_data.append(D)

		if feature=="log melspectrogram":
			D = librosa.feature.melspectrogram(i)
			feature_data.append(librosa.power_to_db(D))

		if feature=="log power melspectrogram":
			D = librosa.feature.melspectrogram(i)
			feature_data.append(librosa.power_to_db(D**2))

		if feature=="mfcc":
			D = librosa.feature.mfcc(i)
			feature_data.append(D)

		if feature=="mfcc deltas":
			D = librosa.feature.mfcc(i)
			feature_data.append(librosa.feature.delta(D))

		if feature=="mfcc deltas deltas":
			D = librosa.feature.mfcc(i)
			feature_data.append(librosa.feature.delta(D,order=2))

		if feature=="chroma":		
			D = librosa.feature.chroma_stft(i)
			feature_data.append(D)

		if feature=="spectral contrast":		
			D = np.abs(librosa.stft(i))   
			contrast = librosa.feature.spectral_contrast(S=D)
			feature_data.append(contrast)

		if feature=="tonnetz":		
			D = librosa.effects.harmonic(i)
			tonnetz = librosa.feature.tonnetz(y=D)
			feature_data.append(tonnetz)


	input_to_clas=[]
	count=0
	for j in feature_data:
		count+=1

		x=cv2.resize(j,(Width,Height))

		#Reverse image
		reversed_x=[]
		for i in x:
			reversed_x.insert(0,i)

		# -------------------------------- normalize --------------------------------
		reversed_x=np.array(reversed_x)
		reversed_x = (reversed_x - reversed_x.min()) * (1/(reversed_x.max() - reversed_x.min()) * 255)

		# plt.figure(figsize=(5,5))
		# plt.imshow(reversed_x,cmap='gray_r')
		# plt.show()

		input_to_clas.append(reversed_x)

		# plt.figure(figsize=(5,5))
		# plt.imshow(reversed_x,cmap='gray_r')
		# plt.show()


	plt.figure(figsize=(5,5)) 
	plt.imshow(input_to_clas[200],cmap='gray_r')
	plt.show()

	plt.figure(figsize=(5,5)) 
	plt.imshow(input_to_clas[800],cmap='gray_r')
	plt.show()

	plt.figure(figsize=(5,5)) 
	plt.imshow(input_to_clas[1200],cmap='gray_r')
	plt.show()

	plt.figure(figsize=(5,5)) 
	plt.imshow(input_to_clas[1800],cmap='gray_r')
	plt.show()

	plt.figure(figsize=(5,5)) 
	plt.imshow(input_to_clas[2300],cmap='gray_r')
	plt.show()

	print(len(input_to_clas))


	with open(pickletowrite,"wb") as f:
		pickle.dump(input_to_clas,f)
