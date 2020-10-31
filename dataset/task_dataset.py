import os
import glob
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import music21 as m21
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset
from dataset.data_utils import encode, decode, get_vocab
from multiprocessing import Pool


class TaskHandler():
    def __init__(self,
                 tracks='all',
                 num_threads=4,
                 cache_dir='./data/processed'):


        self.tracks = tracks
        self.cache_dir = cache_dir
        self.midi_dir = os.path.join(self.cache_dir, 'lpd/lpd_cleansed/midis_tracks={}'.format(tracks))

        self.song_list = np.load(os.path.join(self.cache_dir, 'info/songs_with_genre_info.npy'), allow_pickle=True)
        self.song_to_genres = np.load(os.path.join(self.cache_dir, 'info/song_to_genres.npy'), allow_pickle=True).item()
        self.genre_to_songs = np.load(os.path.join(self.cache_dir, 'info/genre_to_songs.npy'), allow_pickle=True).item()
        self.all_genres = list(self.genre_to_songs.keys())

        # Brief hack because I donked up and overwrote the pop-rock encodings
        self.all_genres.remove("Pop_Rock")

        # Check to make sure that all of the encodings exist
        if all([os.path.exists(os.path.join(self.cache_dir, 'encodings/{}_encodings.pkl'.format(genre.lower()))) for genre in self.all_genres]):
            self.encodings_by_genre = {}

            pbar = tqdm(self.all_genres)
            for genre in pbar:
                pbar.set_description("Loading {} encodings".format(genre))
                song_to_encoding = pickle.load(open(os.path.join(self.cache_dir, 'encodings/{}_encodings.pkl'.format(genre.lower())), "rb"))
                self.encodings_by_genre[genre] = song_to_encoding

        # Otherwise, construct them from the MIDI files
        else:
            self.encodings_by_genre = {genre: {} for genre in self.all_genres}

            if num_threads > 1:
                with Pool(num_threads) as pool:
                    info_by_midi = list(tqdm(pool.imap(self.encode_midi, self.song_list), desc='Encoding MIDI streams', total=len(self.song_list)))

                for song_name, encoding in tqdm(info_by_midi, desc='Compiling encodings', total=len(info_by_midi)):
                    if song_name is not 'Failed':
                        genres = self.song_to_genres[song_name]
                        for genre in genres:
                            self.encodings_by_genre[genre][song_name] = encoding

            else:
                for filename in tqdm(self.song_list, desc='Encoding MIDI streams', total=len(self.song_list)):
                    song_name, encoding = self.encode_midi(filename)
                    if song_name is not 'Failed':
                        genres = self.song_to_genres[song_name]
                        for genre in genres:
                            self.encodings_by_genre[genre][song_name] = encoding

            for genre in self.all_genres:
                genre_encodings - self.encodings_by_genre[genre]
                pickle.dump(genre_encodings, open(os.path.join(self.cache_dir, "encodings/{}_encodings.pkl".format(genre.lower())), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def encode_midi(self, filename):
        path = os.path.join(self.midi_dir, "{}-{}.mid".format(filename, self.tracks))
        try:
            stream = m21.converter.parse(path)
            encoding, _ = encode(stream)

            return filename, encoding
        except:
            return "Failed", []


    def sample_task(self, k_train=4, k_test=1, context_len=12, test_prefix_len=12):
        '''
        Samples a set of k_train examples from a particular genre (meta-training set), as well
        as k_test additional example from the same genre (meta-test). 

        The algorithm will be evaluated on its abiliy to predict the k_test test examples, given 
        some preceding context, after training on the k_train examples


        NOTES: we have a split in genres between train/val/test
        '''
        genre = random.choice(self.all_genres)
        relevant_encodings = self.encodings_by_genre[genre]

        song_names = random.sample(self.genre_to_songs[genre], k=k_train+k_test)

        train_songs, test_songs = song_names[:k_train], song_names[k_train:]

        train_context = np.zeros((k_train, context_len))
        for idx, song in enumerate(train_songs):
            full_encoding = relevant_encodings[song]
            encoding_num_notes = len(full_encoding) // 3
            context_num_notes = context_len // 3

            context_start = 3 * random.randint(0, encoding_num_notes-context_num_notes)
            context = full_encoding[context_start:context_start+context_len]

            train_context[idx, :] = context

        test_context = np.zeros((k_test, test_prefix_len+context_len))
        for idx, song in enumerate(test_songs):
            full_encoding = relevant_encodings[song]
            encoding_num_notes = len(full_encoding) // 3
            context_num_notes = (test_prefix_len+context_len) // 3

            context_start = 3 * random.randint(0, encoding_num_notes-context_num_notes)
            context = full_encoding[context_start:context_start+context_len+test_prefix_len]

            test_context[idx, :] = context

        return torch.tensor(train_context, dtype=torch.long), torch.tensor(test_context, dtype=torch.long), genre



if __name__ == '__main__':
    taskhandler = TaskHandler()

    tr, ts, gr = taskhandler.sample_task()
    print("Genre: ", gr)
    print("Train context:\n", tr)

    # jazz_encodings = taskhandler.encodings_by_genre['Jazz']
    # example_encoding = list(jazz_encodings.values())[5]
    # print(list(jazz_encodings.keys())[5])

    # decoded = decode(example_encoding)
    # decoded.write('midi', 'test.mid')