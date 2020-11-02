# File: utils
# -----------
# Contains general utility functions.

import os
import random
import logging
import torch
import numpy as np

def initialize_experiment(experiment_name, log_name, seed):
    """
    Initializes experiment folders, files and sets the random
    seed, if any is provided.
    """
    # Initialize seeds
    initialize_seeds(seed)

    # Initialize folders
    initialize_experiment_folders(experiment_name)

    # Initialize logging
    initialize_log(experiment_name, log_name)

def initialize_seeds(seed):
    """
    Initializes the seed for (almost) every possible module where
    randomness can occur.
    """
    if seed != -1:
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def initialize_experiment_folders(experiment_name):
    """
    Initializes the folders for the experiment.
    """
    
    # Initialize the main folder
    path = os.path.join("experiments", experiment_name)
    mkdir(path)

    # Initialize subfolders
    mkdir(get_plot_folder(experiment_name))
    mkdir(get_checkpoint_folder(experiment_name))
    mkdir(os.path.join(path, "logs"))

def get_plot_folder(experiment_name):
    """
    Returns the plot folder associated with the experiment name.
    """
    return os.path.join("experiments", experiment_name, "plots")

def get_checkpoint_folder(experiment_name):
    """
    Returns the plot folder associated with the experiment name.
    """
    return os.path.join("experiments", experiment_name, "checkpoints")

def initialize_log(experiment_name, log_name):
    """
    Initialize logging behavior.
    """
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join("experiments", experiment_name, "logs", log_name + ".log")
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

def compute_loss(targets, preds):
    '''
    Computes the cross entropy loss
    '''
    # TO-DO
    pass

def mkdir(dirpath):
    if not os.path.exists(dirpath):                                                     
        try:                                                                                
            os.makedirs(dirpath)                                                       
        except FileExistsError:                                                             
            pass
