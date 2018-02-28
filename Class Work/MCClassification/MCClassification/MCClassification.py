import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#Function to read the MNIST dataset along with the class labels
def readMNISTData():
    """read_data_sets returns a nested structure of python type objects to each 
    component of an element of the dataset"""
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)  #will use MSB of 16bit in TF to describe classes
    """The category labels or variables are represented using the one-hot vector format, where the vector is all-zero
    apart from one element:
    Examples: Catergory 4: 00001000
                Category 2: 00100000
                Category 0: 10000000"""
    #break the data in to training and test sets 
    train_X,train_Y,test_X,test_Y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
    return train_X,train_Y,test_X,test_Y

#Plotting the handwritten digits
def PlotDigits(data,DigitCount):
    #subplots(nrows,ncols,
    f,a = plt.subplots(1,10,figsize = (10,2))
    for i in range(DigitCount):
        a[i].imshow(np.reshape(data[i],(28,28))) #each digit is 28x28 pixel matrix
    plt.show()

#Weight and Biases
def WeightsAndBiases(FeatureDim,ClassDim):
    X = tf.placeholder(tf.float32,[None,FeatureDim])
    Y = tf.placeholder(tf.float32,[None,ClassDim]) #class dim should be 10
    w = tf.Variable(tf.random_normal([]))
    

#Forward Pass
def ForwardPass(w,b,X):
    ...

#Cost function for softmax activation
def MultiClassCost(output,Y):
    ...

#Initialization
def Init():
    ...

#Training Operation
def TrainingOp(LearningRate,mcCost):
    ...

if __name__ == '__main__':
    TrainX,TrainY,TestX,TestY = readMNISTData()
    print("The dimension of each training pattern: ",TrainX.shape[1]) #28x28=764
    print("The number of training patterns: ",TrainX.shape[0])
    print("The dimension of each test pattern: ",TestX.shape[1])
    print("The number of test patterns: ",TestX.shape[0])

    print("The class labels of Training Patterns: ",TrainY[0:10])
    print("The number of class labels of Training Patterns: ",TrainY.shape)

    print("The number of class labels of Test Patterns: ",TestY.shape)
    PlotDigits(TrainX,10)
