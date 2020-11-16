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
from constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
from torch.utils.data import Dataset

class BaselineDataset(Dataset):
    def __init__(self,
                 seq_len=120,
                 tracks='all-no_drums',
                 split='train',
                 cache_dir='./data/processed',
                 return_genre=False):

        self.seq_len = seq_len
        self.tracks = tracks
        self.cache_dir = cache_dir
        self.encodings_dir = os.path.join(self.cache_dir, 'encodings/lpd/lpd_cleansed/midis_tracks={}'.format(tracks))

        self.split = split
        self.train_genres = TRAIN_SPLIT
        self.val_genres = VAL_SPLIT
        self.test_genres = TEST_SPLIT

        self.train_token_ids = []
        self.val_token_ids = []
        self.test_token_ids = []

        self.train_genres_ids = []
        self.val_genres_ids = []
        self.test_genres_ids = []

        self.return_genre = return_genre

        assert all([os.path.exists(os.path.join(self.encodings_dir, '{}_encodings.pkl'.format(genre.lower()))) for 
            genre in self.train_genres + self.val_genres + self.test_genres]),  "Baseline dataset assumes encodings already exist, and at least one encoding is missing!"

        # The baseline dataset compiles all of the songs together, regardless of genre
        pbar = tqdm(self.train_genres)
        for g, genre in enumerate(pbar):
            pbar.set_description("Compiling {} encodings into TRAIN dataset".format(genre))
            song_to_encoding = pickle.load(open(os.path.join(self.encodings_dir, '{}_encodings.pkl'.format(genre.lower())), "rb"))

            for encoding in song_to_encoding.values():
                self.train_token_ids += encoding
                self.train_genres_ids += [g] * len(encoding)

        pbar = tqdm(self.val_genres)
        for g, genre in enumerate(pbar):
            pbar.set_description("Compiling {} encodings into VAL dataset".format(genre))
            song_to_encoding = pickle.load(open(os.path.join(self.encodings_dir, '{}_encodings.pkl'.format(genre.lower())), "rb"))

            for encoding in song_to_encoding.values():
                self.val_token_ids += encoding
                self.val_genres_ids += [g] * len(encoding)

        pbar = tqdm(self.test_genres)
        for g, genre in enumerate(pbar):
            pbar.set_description("Compiling {} encodings into TEST dataset".format(genre))
            song_to_encoding = pickle.load(open(os.path.join(self.encodings_dir, '{}_encodings.pkl'.format(genre.lower())), "rb"))

            for encoding in song_to_encoding.values():
                self.test_token_ids += encoding
                self.test_genres_ids += [g] * len(encoding)

    def train(self):
        self.split = "train"

    def val(self):
        self.split = "val"

    def test(self):
        self.split = "test"

    def __len__(self):
        if self.split == "train":
            return len(self.train_token_ids)//self.seq_len
        elif self.split == "val":
            return len(self.val_token_ids)//self.seq_len
        elif self.split == "test":
            return len(self.test_token_ids)//self.seq_len

    def get_genre(self, genre_per_token):
        """
        Returns the most common genre represented by a sample.
        """
        most_freq = max(set(genre_per_token), key=genre_per_token.count)
        if self.split == "train":
            return self.train_genres[most_freq]
        elif self.split == "val":
            return self.val_genres[most_freq]
        elif self.split == "test":
            return self.test_genres[most_freq]

    def __getitem__(self, idx):

        start = idx * self.seq_len
        if not self.return_genre:

            if self.split == "train":
                return torch.tensor(self.train_token_ids[start:start+self.seq_len], dtype=torch.long)
            elif self.split == "val":
                return torch.tensor(self.val_token_ids[start:start+self.seq_len], dtype=torch.long)
            elif self.split == "test":
                return torch.tensor(self.test_token_ids[start:start+self.seq_len], dtype=torch.long)

        else:

            # very janky, but we need the genre info for sampling (qual eval)
            if self.split == "train":
                return torch.tensor(self.train_token_ids[start:start+self.seq_len], dtype=torch.long), self.get_genre(self.train_genres_ids[start:start+self.seq_len])
            elif self.split == "val":
                return torch.tensor(self.val_token_ids[start:start+self.seq_len], dtype=torch.long), self.get_genre(self.val_genres_ids[start:start+self.seq_len])
            elif self.split == "test":
                return torch.tensor(self.test_token_ids[start:start+self.seq_len], dtype=torch.long), self.get_genre(self.test_genres_ids[start:start+self.seq_len])

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = BaselineDataset(tracks="all-no_drums", return_genre=True)

    dataloader = DataLoader(dataset)
    print("Len dataset in train:", len(dataloader.dataset))
    dataloader.dataset.val()
    print("Len dataset in val:", len(dataloader.dataset))
    dataloader.dataset.test()
    print("Len dataset in test: ", len(dataloader.dataset))