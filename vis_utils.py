# File: vis_utils
# ---------------
# Contains visualization utility functions.

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses, val_iterations, title, xlabel,
                ylabel, folder, name):
    '''
    Plots losses versus iterations.
    '''
    # Plot
    iterations = [i * val_iterations for i in range(1, len(losses) + 1)]
    plt.plot(val_iterations, losses)

    # Set titles
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save figure
    path = os.path.join(folder, name)
    plt.savefig(path)
