#Perceptron using Hebb's rule

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    #Define Training Data
    X = np.array([
        [-2,4],
        [4,1],
        [1,6],
        [2,4],
        [6,2],
        ])

    bias_input = -1
    #Augmented Input
    X = np.array([
        [-2,4,bias_input],
        [4,1,bias_input],
        [1,6,bias_input],
        [2,4,bias_input],
        [6,2,bias_input],
        ])

    print(X)




