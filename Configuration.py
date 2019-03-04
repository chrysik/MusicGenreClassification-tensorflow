def config():
	Hz = 22050
	chunk_len_sec = 5*1000       #miliseconds=1/4second # 10*1000seconds*miliseconds
	Width = 128
	Height = 128
	file_types = ["/*.mp3","/*.flac"]
	Genre_name = ["Classical","Electronic","Hip-hop","Pop","Rock"]
	current_file_path = "H:/clasification-with-pictures"
	return Hz, chunk_len_sec, Width, Height, file_types, Genre_name, current_file_path