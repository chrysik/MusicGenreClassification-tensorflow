from Configuration import config
from Train_Dataset_Functions import create_chunks,chunks_into_nparrays,compute_feature
from Train_Dataset_Functions_2 import create_track_paths,create_labels,input_to_clasif,output_to_clas
from sklearn.model_selection import train_test_split
import pydub


pydub.AudioSegment.ffmpeg = "C:\\ffmpeg\\bin\\ffprobe.exe\\"
Hz, chunk_len_sec, Width, Height, file_types, Genre_name, current_file_path = config()

# # Create track paths
# genre_paths=create_track_paths(Genre_name,file_types,current_file_path,"/Dataset_CEHPR_500/")
# # Save labels
# labels=create_labels(genre_paths)
# # Save waveforms
# chunks_into_nparrays(current_file_path,genre_paths,"dataset_CEHPR_500_waveform.pkl",Hz)

compute_feature("dataset_CEHPR_500_waveform.pkl","dataset_CEHPR_500_tonnetz.pkl",Width,Height,feature="tonnetz")
