"""
Parses the raw format of the Lakh MIDI dataset as provided by the Music
and AI Lab at Academica Sinica (which are actually in a directory-dense
.npz [pypianoroll] format). This converts each pypianroll object into a
valid .midi file and extracts the selected tracks from the overall songs.
(e.g. can be used to extract only the bass tracks for a bass dataset).

Usage:
======
--tracks        - Specify the tracks to extract to .midi file
                - Default: 'all'
                - Options: ['all', 'Strings', 'Bass', 'Drums', 'Guitar', 'Piano']

--raw_data_dir  - Base dir to the Lakh MIDI dataset (e.g. './PATH_TO/lpd_cleansed')
                - Default: './data/raw/lpd/lpd_cleansed'
    
--dest_dir      - Base dir to where the output MIDIs should be stored
                - Default: './data/processed/lpd/lpd_cleansed'      



Assumes the Lakh MIDI dataset (lpd_5_cleansed) is in ./data_raw. Will
output the extracted MIDI files to
./data_processed/midis_tracks=<extracted_tracks>
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pypianoroll
from pypianoroll import Multitrack
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import functools

def process_track_path(path, selected_tracks, collection_dir):
    for checksum in os.listdir(path):
        load_dir = os.path.join(path, checksum)
        multiroll = Multitrack(load_dir)

        # Remove all tracks but those in selected_tracks
        if "all" not in selected_tracks:

            to_remove = [idx for idx, track in enumerate(multiroll.tracks) \
                            if track.name not in selected_tracks]
            multiroll.remove_tracks(to_remove)

            # Make sure our selected tracks persist
            # assert len(multiroll.tracks) == len(selected_tracks)
            if len(multiroll.tracks) != len(selected_tracks):
                continue

        # e.g. save_name = TR#########-bass-piano.mid
        name = os.path.basename(path)
        save_name = '{}-{}.mid'.format(name, "-".join(selected_tracks).lower())
        save_path = os.path.join(collection_dir, save_name)
        multiroll.write(save_path)

def collect_midis(base_dir, collection_dir, selected_tracks=["all"]):
    """
    Collects .npz files from raw data into processed data folders as .mid
    - selected_track should be a list of track(s) 
        > options: ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings', 'all']
    """

    if not os.path.exists(collection_dir):
        print("Creating collection directory %s" % collection_dir)
        os.mkdir(collection_dir)

    selected_tracks.sort()  # to keep consistency in filename later

    # Find all of the track name directories
    track_paths = list(Path(base_dir).rglob('TR*'))

    track_paths = track_paths[:6*20]

    # create partial function with params pre-loaded so we can use the worker pool
    # note: this can't be a lambda function since those aren't pickle-able
    _foo = functools.partial(process_track_path, selected_tracks=selected_tracks, 
                                                 collection_dir=collection_dir)
    with Pool(6) as p:
        _ = list(tqdm(p.imap(_foo, track_paths), total=len(track_paths)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', default='all', choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--base_data_dir',type=str, default="./data/raw/lpd/lpd_cleansed")
    parser.add_argument('--dest_dir',type=str, default="./data/processed/lpd/lpd_cleansed")

    args = parser.parse_args()
    if args.tracks == 'all':
        args.tracks = ['all']

    # correct path sep for the os
    args.base_data_dir = os.path.normpath(args.base_data_dir)
    args.dest_dir = os.path.normpath(args.dest_dir)

    print(args.base_data_dir)
    
    full_collection_dir = os.path.join(args.dest_dir, 'midis_tracks=' + '-'.join(args.tracks))

    print("Collecting MIDI files (tracks = {})".format(args.tracks))
    collect_midis(args.base_data_dir, full_collection_dir, args.tracks)
