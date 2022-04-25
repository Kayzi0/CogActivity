import numpy as np
import os
import helpers
from matplotlib import pyplot as plt
from scipy.signal import decimate

labels = helpers.load_data("trainLabels.npy", "train")

gravity = helpers.load_data("trainGravity.npy", "train")
magnet = helpers.load_data("trainMagnetometer.npy", "train")
linAcc = helpers.load_data("trainLinearAcceleration.npy", "train")
print(magnet.shape)
print(gravity.shape)
print(linAcc.shape)
x1 = np.arange(0,800,1)
x2 = np.arange(0,200,1)
y = gravity[0,:,0]


dec = decimate(y, 4)
print(dec.shape)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(x1, y)

ax1.set_title("Plot of original data")
ax2.plot(x2,dec)
ax2.set_title("Plot of decimated data")

plt.show()
