import os
import csv
import sys
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

class MaestroDataset(Dataset):
    def __init__(self,
                 split="train",
                 context_len=120,
                 meta=True,
                 k_train=4,
                 k_test=1,
                 num_threads=4,
                 cache_dir='./data/processed',
                 raw_data_dir='./data/raw'):


        self.split = split
        self.context_len = context_len
        self.meta = meta
        self.k_train = k_train
        self.k_test = k_test

        self.midi_dir = os.path.join(raw_data_dir, 'maestro-v2.0.0')
        self.encodings_dir = os.path.join(cache_dir, 'encodings/maestro')


        if all([os.path.exists(os.path.join(self.encodings_dir, "{}_encodings.pkl".format(split))) for split in ["train", "val", "test"]]):
            
            self.train_titles = pickle.load(open(os.path.join(self.encodings_dir, "train_titles.pkl"), "rb"))
            self.train_encodings = pickle.load(open(os.path.join(self.encodings_dir, "train_encodings.pkl"), "rb"))

            self.val_titles = pickle.load(open(os.path.join(self.encodings_dir, "val_titles.pkl"), "rb"))
            self.val_encodings = pickle.load(open(os.path.join(self.encodings_dir, "val_encodings.pkl"), "rb"))

            self.test_titles = pickle.load(open(os.path.join(self.encodings_dir, "test_titles.pkl"), "rb"))
            self.test_encodings = pickle.load(open(os.path.join(self.encodings_dir, "test_encodings.pkl"), "rb"))

        else:
            self.filename_to_title = {}
            self.title_to_split = {}

            # These lists hold all the titles for each split, even if they might later fail to be parsed
            all_train_titles, all_val_titles, all_test_titles = [], [], []

            with open(os.path.join(self.midi_dir, "maestro-v2.0.0.csv"), "r") as file:
                reader = csv.reader(file)
                for line in reader:
                    if line[0] == 'canonical_composer':
                        continue

                    canonical_composer, canonical_title, split, _, full_filename, _, _ = line
                    filename = full_filename.split("/")[1]

                    title = ""
                    composers = canonical_composer.split("/")
                    for composer in composers:
                        lastname = composer.strip().split(" ")[-1].upper()
                        title += lastname + "-"

                    songname = "_".join([word for word in canonical_title.replace(".", "").replace(",", "").split(" ") if word != ""])
                    title += songname

                    repeat_idx = 1
                    while title + "_" + str(repeat_idx) in self.filename_to_title.values():
                        repeat_idx += 1

                    title += "_" + str(repeat_idx)

                    self.filename_to_title[filename] = title
                    self.title_to_split[title] = split

                    if split == "train":
                        all_train_titles.append(title)
                    elif split == "val":
                        all_val_titles.append(title)
                    elif split == "test":
                        all_test_titles.append(title)

            # In contrast, these lists hold only the titles for which the corresponding MIDI was successfully parsed
            self.train_titles, self.val_titles, self.test_titles = [], [], []

            # These lists hold the encodings for each song, in the same order as self.train/val/test_titles
            self.train_encodings, self.val_encodings, self.test_encodings = [], [], []

            all_songs = glob.glob(os.path.join(self.midi_dir, "*/*.midi"))
            
            if num_threads > 1:
                with Pool(num_threads) as pool:
                    titles_encodings = list(tqdm(pool.imap(self.encode_midi, all_songs), desc="Encoding MIDI streams", total=len(all_songs)))

                    fail_count = 0
                    for title, encoding in titles_encodings:
                        if title == "Failed":
                            fail_count += 1
                            continue

                        split = self.title_to_split[title]

                        if split == "train":
                            self.train_titles.append(title)
                            self.train_encodings.append(encoding)

                        elif split == "validation":
                            self.val_titles.append(title)
                            self.val_encodings.append(encoding)

                        elif split == "test":
                            self.test_titles.append(title)
                            self.test_encodings.append(encoding)

            else:
                fail_count = 0
                # for path in tqdm(all_songs, desc="Encoding MIDI streams", total=len(all_songs)):
                for idx, path in enumerate(all_songs):
                    print("\n[{}/{}]".format(idx, len(all_songs)))
                    title, encoding = self.encode_midi(path)

                    if title == "Failed":
                        fail_count += 1
                        continue

                    split = self.title_to_split[title]

                    if split == "train":
                        self.train_titles.append(title)
                        self.train_encodings.append(encoding)

                    elif split == "validation":
                        self.val_titles.append(title)
                        self.val_encodings.append(encoding)

                    elif split == "test":
                        self.test_titles.append(title)
                        self.test_encodings.append(encoding)

            print("Successfully encoded MIDIs, with {} failures".format(fail_count))

            # Dump the title list and encoding list for each split
            pickle.dump(self.train_titles, open(os.path.join(self.encodings_dir, "train_titles.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train_encodings, open(os.path.join(self.encodings_dir, "train_encodings.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

            pickle.dump(self.val_titles, open(os.path.join(self.encodings_dir, "val_titles.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.val_encodings, open(os.path.join(self.encodings_dir, "val_encodings.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

            pickle.dump(self.test_titles, open(os.path.join(self.encodings_dir, "test_titles.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.test_encodings, open(os.path.join(self.encodings_dir, "test_encodings.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # If we are not in META mode, then we collapse all the encodings from a given split into one list
        if self.meta == False:
            self.train_encodings = sum(self.train_encodings, [])
            self.val_encodings = sum(self.val_encodings, [])
            self.test_encodings = sum(self.test_encodings, [])

        # These dictionaries just help clean up references to the current split's encodings / titles
        self.encodings_dict = {"train": self.train_encodings,
                               "val": self.val_encodings,
                               "test": self.test_encodings}

        self.titles_dict = {"train": self.train_titles,
                            "val": self.val_titles,
                            "test": self.test_titles}

    def encode_midi(self, path):
        '''
        Encodes the MIDI file at 'path' as a sequence of (pitch, duration, advance) tokens
        '''
        try:
            print("Parsing stream at:", path)
            stream = m21.converter.parse(path)

            print("\tRunning encoding")
            encoding = encode(stream)
            print("\tFinished!")

            # Extact just the MIDI filename from the full path
            filename = path.split("/")[-1]
            title = self.filename_to_title[filename]

            return title, encoding

        except:
            print("Error:", sys.exc_info())
            return "Failed", []

    def train(self):
        self.split = "train"

    def val(self):
        self.split = "val"

    def test(self):
        self.split = "test"

    def get_non_overlapping_subsequences(self, sequence, N, K):
        '''
        From Armin Rigo at https://stackoverflow.com/questions/18641272/n-random-contiguous-and-non-overlapping-subsequences-each-of-length

        N: number of subsequences
        K: length of each subsequence
        '''
        indices = range(len(sequence) - (K - 1) * N)
        result = []
        offset = 0

        for i in sorted(random.sample(indices, N)):
            i += offset
            result.append(sequence[i:i+K])
            offset += K - 1

        return result

    def __len__(self):
        '''
        If meta=True, then the length is the number of songs in the current split
        If meta=False, then the length is the number of sequences of size 'context_len' that fit into the current split
        '''
        if self.meta == True:
            return len(self.encodings_dict[self.split])
        else:
            return len(self.encodings_dict[self.split]) // self.context_len


    def __getitem__(self, idx):
        '''
        If meta=True, then idx determines the song, from which k_train and k_test non-overlapping samples are taken and returned
        '''
        if self.meta == True:
            support = np.zeros((self.k_train, self.context_len))
            query = np.zeros((self.k_test, self.context_len))

            encoding = self.encodings_dict[self.split][idx]

            # Here a "note index" collapses each tuple of (pitch, duration, advance) into a single value. We're
            # going to extract subsequences from these note indexes, and then expand them when we slice into the
            # encoding
            note_idxs = np.arange(0, len(encoding) // 3)

            note_subsequences = self.get_non_overlapping_subsequences(note_idxs, self.k_train+self.k_test, self.context_len // 3)

            train_subs, test_subs = note_subsequences[:self.k_train], note_subsequences[self.k_train:]
            for i, subsequence in enumerate(train_subs):
                # Here we multiply by 3 to convert this "note index" back into an "encoding index"
                start = 3 * subsequence[0]
                support[i, :] = encoding[start:start+self.context_len]

            for i, subsequence in enumerate(test_subs):
                start = 3 * subsequence[0]
                query[i, :] = encoding[start:start+self.context_len]

            return torch.tensor(support, dtype=torch.long), torch.tensor(query, dtype=torch.long)

        else:
            start = idx * self.context_len
            return torch.tensor(self.encodings_dict[self.split][start:start+self.context_len], dtype=torch.long)

if __name__ == '__main__':
    dataset = MaestroDataset(meta=True, split="train", context_len=30, k_train=1, num_threads=1)

    print("Num train titles:", len(dataset.train_titles))
    print("Num val titles:", len(dataset.val_titles))
    print("Num test titles:", len(dataset.test_titles))


