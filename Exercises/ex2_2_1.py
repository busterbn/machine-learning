import importlib_resources
import numpy as np
from matplotlib.pyplot import cm, figure, imshow, show, subplot, title, xlabel, yticks
from scipy.io import loadmat

filename = importlib_resources.files("dtuimldmtools").joinpath("data/zipdata.mat")
# Index of the digit to display
i = 0

# Load Matlab data file to python dict structure
mat_data = loadmat(filename)

# Extract variables of interest
testdata = mat_data["testdata"]
traindata = mat_data["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]


# Visualize the i'th digit as a vector
f = figure()
subplot(4, 1, 4)
imshow(np.expand_dims(X[i, :], axis=0), extent=(0, 256, 0, 10), cmap=cm.gray_r)
xlabel("Pixel number")
title("Digit in vector format")
yticks([])

# Visualize the i'th digit as an image
subplot(2, 1, 1)
I = np.reshape(X[i, :], (16, 16))
imshow(I, extent=(0, 16, 0, 16), cmap=cm.gray_r)
title("Digit as an image")

show()

print("Ran Exercise 2.2.1")
