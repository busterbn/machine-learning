# Exercise 4.2.7

# requires data from exercise 4.2.1
from ex4_2_1 import *
from matplotlib.pyplot import (
    cm,
    colorbar,
    figure,
    imshow,
    show,
    title,
    xlabel,
    xticks,
    ylabel,
)
from scipy.stats import zscore

X_standarized = zscore(X, ddof=1)

figure(figsize=(12, 6))
imshow(X_standarized, interpolation="none", aspect=(4.0 / N), cmap=cm.gray)
xticks(range(4), attributeNames)
xlabel("Attributes")
ylabel("Data objects")
title("Fisher's Iris data matrix")
colorbar()

show()

print("Ran Exercise 4.2.7")
