from Train_Dataset_Functions import create_chunks,chunks_into_nparrays,compute_feature
from Configuration import config
from Train_Dataset_Functions_2 import create_track_paths,create_labels,input_to_clasif,output_to_clas
from sklearn.model_selection import train_test_split
import pydub

pydub.AudioSegment.ffmpeg = "C:\\ffmpeg\\bin\\ffprobe.exe\\"
Hz, chunk_len_sec, Width, Height, file_types, Genre_name, current_file_path = config()
genre_paths = create_track_paths(Genre_name,file_types,current_file_path,"/Dataset_CEHPR/")
create_chunks(chunk_len_sec,genre_paths)
