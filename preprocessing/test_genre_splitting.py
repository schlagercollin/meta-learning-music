import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

SAVE_PATH = "./data/processed"
# DATA_PATH = "./data/processed/lpd/lpd_cleansed/midis_tracks=all"
DATA_PATH = "./data/processed/lpd/lpd_cleansed/midis_tracks=all-no_drums"
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

genre_to_songs = defaultdict(list)
for track_name in track_names:
    if track_name in song_to_genres:
        genres = song_to_genres[track_name]
        for genre in genres:
            genre_to_songs[genre].append(track_name)


# Print out some stats
print("We have MIDI data for {} songs".format(len(track_names)))
total = 0
for key, tracks in genre_to_songs.items():
    print("\tGenre {} has {} songs".format(key, len(tracks)))
    total += len(tracks)
print("We have genre data for {} ({}%) of our songs".format(total, np.round(100 * total / len(track_names), 2)))

# Save the mappings from song to genre, and vice versa
np.save(os.path.join(SAVE_PATH, "song_to_genres.npy"), song_to_genres)
np.save(os.path.join(SAVE_PATH, "genre_to_songs.npy"), genre_to_songs)

# Save the list of songs for which we have genre info
songs_with_info = sum([genre_to_songs[genre] for genre in genre_to_songs.keys()], [])
np.save(os.path.join(SAVE_PATH, "tracks_with_genre_info.npy"), songs_with_info)

# We want to make TRAIN/VAL/TEST splits by genre, so that we have data for every task
