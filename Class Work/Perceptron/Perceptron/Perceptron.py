#Perceptron using Hebb's Rule

import numpy as np 
#import matplotlib
#from matplotlib import pyplot as plt

if __name__ == '__main__': #how to designate a main in python
    #Define training data
    X = np.array([
        [-2,4],
        [4,1],
        [1,6],
        [2,4],
        [6,2],
        ])

    bias_input = -1 #can be plus or minus 1

    #augmented input

    X=np.array([
        [-2,4,bias_input],
        [4,1,bias_input],
        [1,6,bias_input],
        [2,4,bias_input],
        [6,2,bias_input],])

    #for i in range(size(X))


    print(X)