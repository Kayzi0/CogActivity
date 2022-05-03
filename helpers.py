import os
import numpy as np
from scipy import signal
from sklearn import preprocessing

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
            downsampled_data[i,:,j] = signal.decimate(data[i,:,j], downsample_factor)
    return downsampled_data

def upsample(data, target_size):
    upsampled_data = np.zeros((data.shape[0], target_size, data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            upsampled_data[i,:,j] = signal.resample(data[i,:,j], target_size)
    
    return upsampled_data

def minmaxnormalise(data):
    minmax_data = np.copy(data)
    for i in range(data.shape[2]):
        minmax_data[:,:,i] = preprocessing.minmax_scale(data[:,:,i])
    return minmax_data

def denoise(data):
    denoised_data = np.copy(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            denoised_data[i,:,j] = signal.wiener(data[i,:,j])

    return denoised_data

def preprocess_data(data, sampling_type = "decimate", target_size = 200):
    if sampling_type == "resample":
        out_data = upsample(data, target_size)
    elif sampling_type == "decimate":
        out_data = downsample(data, target_size)

    out_data = denoise(out_data)
    #out_data = minmaxnormalise(out_data)

    return out_data
    