import os
import numpy as np

def load_data(filename, type):
    if type == "test":
        path = "data_raw/testing"
    elif type == "train":
        path = "data_raw/training"
    full_path = os.path.join(path, filename)
    return np.load(full_path)