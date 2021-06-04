import os
import numpy as np


def load_cosmos_galaxies(filename="Images_Filtered_Rotated.npy"):
    home = os.path.expanduser("~")
    path = os.path.join(home, "Data", "galaxy-vae", filename)
    return np.load(path)


def load_benchmark_galaxies(filename="benchmark_images.npy"):
    home = os.path.expanduser("~")
    path = os.path.join(home, "Data", "galaxy-vae", filename)
    return np.load(path)
