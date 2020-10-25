# meta-learning-music
Code repository for our Stanford CS 330 (Deep Multi-task and Meta-Learning) course project

# Dependencies

I've set up a Conda environment to keep track of our dependencies. If you have Conda installed, you can just run

`conda env create -f environment.yml`

in the base directory. This will create a virtual environment called cs330 that we can use. You can activate
the virtual environment by running `conda activate cs330`. 
I'm happy to use another form of environment control if Conda doesn't cut it for us. Maybe Docker since we'll
want to make it easy to push code to our cloud instance?

# Data

https://salu133445.github.io/lakh-pianoroll-dataset/dataset

Download the cleansed Lakh Pianoroll Dataset compressed file and extract it to the `./data/raw` directory.
This should create the folder structure: `./data/raw/lpd/lpd_cleansed/` where the contents of the `lpd_cleansed`
folder is a large subfolder tree that terminates in `.npz` files that correspond to the pianoroll format.
(I have no idea why they choose to store the data in this directory-heavy way. It might be worth collapsing the
dataset down, since it's not a very git-friendly source tree).

These `.npz` files represent the pianoroll tracks, and they can be conveniently manipulated using the `pypianoroll` Python
library. We have been using `pypianoroll` to extract tracks from the music. However, ultimately, we use the output .midi 
files to create the dataset. Therefore, I'm not sure if we really need to use the `pypianoroll` library, and, in turn, the awkward `.npz` file format.

Take a look at the script inside `./preprocessing/collect_midis.py`. It is a simple script that
traverses the raw data directory file tree and processes each pianoroll (selecting a few tracks of interest)
before writing it out to a .midi file. 

# Dataset

I've copied over the dataset class we used for our CS 236 project. You can find it inside `./dataset/midi_sequence_dataset.py`. 
This dataset converts the .midi track into a series of tokens that can be used with a language model. `./dataset/data_utils.py`
contains some helpfer functions that carry out this conversion. It uses the `music21` library out of MIT to aid in this token processing from the .midi format.
