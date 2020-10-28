import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

SAVE_PATH = "./data/processed"
DATA_PATH = "./data/processed/lpd/lpd_cleansed/midis_tracks=all"
GENRE_PATH = "./data/raw/genres/msd_tagtraum_cd1.cls"

# First, we need to collect all the file names in our dataset
track_names = [filename[:filename.find('-')] for filename in os.listdir(DATA_PATH)]


song_to_genres = {}
with open(GENRE_PATH, "r") as file:
    lines = file.readlines()
    for line in tqdm(lines, total=len(lines), desc='Reading genre information'):
        words = [word for word in line.split('\t') if word is not '']
        track_name = words[0]
        genres = [word.replace("\n", "") for word in words[1:]]
        
        song_to_genres[track_name] = genres

genre_mapping = defaultdict(list)
for track_name in track_names:
    if track_name in song_to_genres:
        genres = song_to_genres[track_name]
        for genre in genres:
            genre_mapping[genre].append(track_name)


# Print out some stats
print("We have MIDI data for {} songs".format(len(track_names)))
total = 0
for key, tracks in genre_mapping.items():
    print("\tGenre {} has {} songs".format(key, len(tracks)))
    total += len(tracks)
print("We have genre data for {} ({}%) of our songs".format(total, np.round(100 * total / len(track_names), 2)))
