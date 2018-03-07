"""
David Mohler
EE-5410: Neural Networks
Assignment #3: Problem #1
"""

#Project Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #Library for 3D plotting
import csv  #Python library to read excel csv files

#Function to plot input patterns 
def PlotData(features,labels,PlotTitle):
    fig = plt.figure(PlotTitle) #create a figure object
    ax = fig.add_subplot(111) #create a subplot 
    for i, feature in enumerate(features):
        if labels[i] == 0:
            ax.scatter(feature[0],feature[1],s = 120, color = 'red', marker = "_", linewidth = 2)
        else:
            ax.scatter(feature[0],feature[1],s = 120, color = 'blue', marker = "+", linewidth = 2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()


"""MAIN PROGRAM BODY"""
#-------------------------------------------------------------------------------#
if __name__ == '__main__': 
    X_aug = np.array([[3,1,1],
                      [4,0,1],
                      [4,-1,1],
                      [5,2,1],
                      [5,3,1],
                      [3,3,1],
                      [2,0,1],
                      [1,1,1]
                      ])
    Y = np.array([0,0,0,1,1,1,1,1]) #Class labels for input data
    bias =  X_aug[0,2] 
    print("bias: ",bias) 
    print("Augmented Input Data: \n",X_aug) 
    PlotData(X_aug,Y,"Input Patterns")