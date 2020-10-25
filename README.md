# meta-learning-music
Code repository for our Stanford CS 330 (Deep Multi-task and Meta-Learning) course project

# Data

Download the cleansed Lakh Pianoroll Dataset compressed file and extract it to the `./data/raw` directory.
This should create the folder structure: `./data/raw/lpd/lpd_cleansed/` where the contents of the `lpd_cleansed`
folder is a large subfolder tree that terminates in `.npz` files that correspond to the pianoroll format.
(I have no idea why they choose to store the data in this directory-heavy way. It might be worth collapsing the
dataset down, since it's not a very git-friendly source tree).

These `.npz` files represent the pianoroll tracks, and they can be manipulated using the `pypianoroll` Python
library. Take a look at the script inside `./preprocessing/collect_midis.py`. It is a simple script that
traverses the raw data directory file tree and processes each pianoroll (selecting a few tracks of interest)
before writing it out to a .midi file. 
