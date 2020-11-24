# meta-learning-music

Code repository for our Stanford CS 330 (Deep Multi-task and Meta-Learning) course project

<img src="https://github.com/schlagercollin/meta-learning-music/blob/main/images/muml.png" alt="drawing" width="400"/>

<img src="https://raw.githubusercontent.com/schlagercollin/meta-learning-music/main/images/sample.png" alt="drawing" width="700"/>

# Dependencies

Dependencies can be easily installed by replicating the provided Conda environment. Execute

`conda env create -f environment.yml`

in the base directory to create this environment. Note that this environment was used on Windows10. 
The environment can be activated using `conda activate cs330`.

# Data

Datasets have not been included due to their size. See the following links for download of the Lakh Midi Dataset and the Maestro Dataset (both used for this project).

https://salu133445.github.io/lakh-pianoroll-dataset/dataset

https://magenta.tensorflow.org/datasets/maestro

Raw datasets should be extracted to the `./data/raw` directory.
This should create the folder structure: `./data/raw/lpd/lpd_cleansed/` where the contents of the `lpd_cleansed`

See the preprocessing scripts in `./preprocessing` to convert the raw data into data usable for this project.
The `./dataset` directory also includes the dataset classes for creating token embeddings for use with the language models.
