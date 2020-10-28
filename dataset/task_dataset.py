import os
import glob
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import music21 as m21
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset
from dataset.data_utils import encode, get_vocab
from multiprocessing import Pool


class TaskHandler():
    def __init__(self,
                 tracks='all',
                 num_threads=4,
                 cache_dir='./data/processed/lpd/lpd_cleansed'):


        self.data_dir = os.path.join(cache_dir, 'midis_tracks={}'.format(tracks))
        self.tracks = tracks

        tracks_with_info = np.load("./data/processed/tracks_with_genre_info.npy", allow_pickle=True)
        
        self.encodings = {}
        if num_threads > 1:
            with Pool(num_threads) as pool:
                info_by_midi = list(tqdm(pool.imap(self.encode_midi, tracks_with_info), desc='Encoding MIDI streams', total=len(tracks_with_info)))

            for filename, encoding in tqdm(info_by_midi, desc='Compiling encodings', total=len(info_by_midi)):
                if filename is not "Failed":
                    self.encodings[filename] = encoding

        else:
            for filename in tqdm(tracks_with_info, desc='Encoding MIDI streams', total=len(tracks_with_info)):
                result_filename, encoding = self.encode_midi(filename)
                if result_filename is not 'Failed':
                    self.encodings[filename] = encoding                    

        np.save('./data/processed/track_encodings.npy', self.encodings)
        print("Number of successful encodings:", len(self.encodings))


    def encode_midi(self, filename):
        path = os.path.join(self.data_dir, "{}-{}.mid".format(filename, self.tracks))
        try:
            stream = m21.converter.parse(path)
            encoding, _ = encode(stream)

            return filename, encoding
        except:
            return "Failed", []

if __name__ == '__main__':
    taskhandler = TaskHandler()