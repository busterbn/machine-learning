# exercise 4.1.6

import importlib_resources
import numpy as np
import scipy.linalg as linalg
from matplotlib.pyplot import cm, figure, imshow, show, subplot, title, xticks, yticks
from scipy.io import loadmat

filename = importlib_resources.files("dtuimldmtools").joinpath("data/zipdata.mat")
# Digits to include in analysis (to include all: n = range(10))
n = [1]

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat(filename)["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]
N, M = X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = y == v
    class_mask = class_mask | cmsk
X = X[class_mask, :]
y = y[class_mask]
N = np.shape(X)[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

figure()
subplot(1, 2, 1)
I = np.reshape(mu, (16, 16))
imshow(I, cmap=cm.gray_r)
title("Mean")
xticks([])
yticks([])
subplot(1, 2, 2)
I = np.reshape(s, (16, 16))
imshow(I, cmap=cm.gray_r)
title("Standard deviation")
xticks([])
yticks([])

show()

print("Ran Exercise 4.1.6")
