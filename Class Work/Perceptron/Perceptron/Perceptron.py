#Perceptron using Hebb's Rule

import numpy as np 
import matplotlib
from matplotlib import pyplot as plt


#when running python program, this line prevents use in an imported module (i.e. cna't import this to another .py and use whats in __main__) 
if __name__ == '__main__': #how to designate a main in python
    #Define training data
    X = np.array([
        [-2,4],
        [4,1],
        [1,6],
        [2,4],
        [6,2],
        ])

    bias_input = -1 #plus or minus 1

    #augmented input (in class) 
    #X=np.array([
    #    [-2,4,bias_input],
    #    [4,1,bias_input],
    #    [1,6,bias_input],
    #    [2,4,bias_input],
    #    [6,2,bias_input],])


    #Loop augmentation of data with bias (more dynamically coded) 
    augment_data = np.zeros((X.shape[0],X.shape[1]+1)) #preallocate temp array to augment with bias value
    for i in range(len(X)):
        augment_data[i] = np.append(X[i],bias_input)
    X = augment_data

    #To get back original data
    #X = np.delete(X,X.shape[1]-1,1) #strips the bias off and returns original data array (data,index of bias,axis) 
    
    print(X)