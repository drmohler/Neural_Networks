"""
David R Mohler
EE 5410: Neural Networks
Exam #1: Problem 1

Classification of handwritten MNIST data with backpropagation
"""

#Library Imports
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

#------------------------ Functions ----------------------------#

def readMNIST(): 
    mnist = input_data.read_data_sets("MNIST_data/")
    train_X,train_Y,test_X,test_Y = mnist.train.images,mnist.train.labels, mnist.test.images,mnist.test.labels

    return train_X,train_Y,test_X,test_Y

def PlotDigits(data,DigitCount):
    #subplots(nrows,ncols,
    f,a = plt.subplots(1,10,figsize = (10,2))
    for i in range(DigitCount):
        a[i].imshow(np.reshape(data[i],(28,28)))
    plt.show()


#-------------------- Main Implementation-----------------------#

if __name__=="__main__":

    TrainX,TrainY,TestX,TestY = readMNIST()
    
    print("The dimension of each training pattern: ",TrainX.shape[1])    # returns 784 ( = 28x28, in linear space)
    print("The number of training patterns: ",TrainX.shape[0])           # 55000 training patters
    print("The dimension of each test pattern: ",TestX.shape[1])         # 784
    print("The number of test patterns: ",TestX.shape[0])                #10,000

    print("The class labels of Training Patterns: ",TrainY[0:10])
    print("The number of class labels of Training Patterns: ",TrainY.shape)

    print("The number of class labels of Test Patterns: ",TestY.shape)
    PlotDigits(TrainX,10)

