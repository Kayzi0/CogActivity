import os
import numpy as np
from scipy.signal import decimate

def load_data(filename, type):
    if type == "test":
        path = "data_raw/testing"
    elif type == "train":
        path = "data_raw/training"
    full_path = os.path.join(path, filename)
    return np.load(full_path)

def downsample(data, target_size):
    downsample_factor = int(data.shape[1]/target_size)
    downsampled_data = np.zeros((data.shape[0], target_size, data.shape[2]))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            downsampled_data[i,:,j] = decimate(data[i,:,j], downsample_factor)
    return downsampled_data
