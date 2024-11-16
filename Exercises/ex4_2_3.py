# Exercise 4.2.3

# requires data from exercise 4.2.1
from ex4_2_1 import *
from matplotlib.pyplot import boxplot, show, title, xticks, ylabel

boxplot(X)
xticks(range(1, 5), attributeNames)
ylabel("cm")
title("Fisher's Iris data set - boxplot")
show()

print("Ran Exercise 4.2.3")
