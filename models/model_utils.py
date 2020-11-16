# File: model_utils
# -----------------
# Utility functions for initializing, saving and
# loading models and maybe more.

import torch
import os

import constants
from models.simple_lstm import SimpleLSTM
from models.simple_transformer import SimpleTransformer


def initialize_model(experiment_name, model_type, load_from_iteration,
                     device, args):
    '''
    Initializes a model of the provided model type. If load_from_iteration
    is not set to -1, then we will load the model from the associated checkpoint
    in the experiment folder named experiment_name
    '''
    # Initialize a completely new model
    model = get_model(model_type, args)
    model = model.to(device)

    # Load parameters from a checkpoint if necessary
    if load_from_iteration != -1:
        load_model(model, experiment_name, load_from_iteration)

    return model

def get_model(model_type, args):
    '''
    Gets the model of the specified model type.
    '''
    if model_type == "SimpleLSTM":
        return SimpleLSTM(args.embed_dim, args.hidden_dim, constants.VOCAB_SIZE)
    elif model_type == "SimpleTransformer":
        return SimpleTransformer(args.embed_dim, args.hidden_dim, args.num_blocks,
                                 args.num_heads, args.context_len-1, constants.VOCAB_SIZE)

def load_model(model, experiment_name, load_from_iteration):
    '''
    Loads model parameters from a checkpoint.
    '''
    path = os.path.join("experiments", experiment_name, "checkpoints",
                        "iter_{}.pth".format(load_from_iteration))

    print(f"Loading model from {path}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} doesn't exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))

def save_model(model, experiment_name, iteration):
    '''
    Saves model parameters to the checkpoint folder
    '''
    path = os.path.join("experiments", experiment_name, "checkpoints",
                        "iter_{}.pth".format(iteration))
    torch.save(model.state_dict(), path)
