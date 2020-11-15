# File: vis_utils
# ---------------
# Contains visualization utility functions.

import os

# Sets backend to non-visual if DISPLAY isn't set up
if os.environ.get("DISPLAY", None) is None:
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses, val_iterations, title, xlabel,
                ylabel, folder, name):
    '''
    Plots losses versus iterations.
    '''
    # Plot
    iterations = [i * val_iterations for i in range(1, len(losses) + 1)]
    plt.plot(iterations, losses)

    # Set titles
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save figure
    path = os.path.join(folder, name)
    plt.savefig(path)


if __name__ == "__main__":
    print("Saving test plot to './test.png'")
    plot_losses([0.9, 0.8, 0.7], 50, "Test Plot", "Iterations", "Loss", "", "test.png")
