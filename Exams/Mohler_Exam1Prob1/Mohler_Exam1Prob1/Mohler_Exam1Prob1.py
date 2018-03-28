"""
David R Mohler
EE 5410: Neural Networks
Exam #1: Problem 1

Classification of handwritten MNIST data with backpropagation
"""

#Library Imports
import random
import math
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


class NeuralNetwork: 
    def __init__(self,numInput,numHidden,numOuput,seed):
        self.ni = numInput #input dimension of MNIST images (should be 784) 
        self.nh = numHidden # number of neurons in hidden layer
        self.no = numOuput #number of neurons in output layer (should be 10 and softmax) 

        #preallocate for all neurons
        self.iNodes = np.zeros(shape=[self.ni],dtype=np.float32) 
        self.hNodes = np.zeros(shape=[self.nh],dtype=np.float32) 
        self.oNodes = np.zeros(shape=[self.no],dtype=np.float32) 

        #allocate for weight matrices
        self.ihWeights = np.zeros(shape=[self.ni,self.nh],dtype=np.float32) 
        self.hoWeights = np.zeros(shape=[self.nh,self.no],dtype=np.float32) 

        #allocate for biases in hidden and output layers
        self.hBiases = np.zeros(shape=[self.nh],dtype=np.float32) #hidden biases
        self.oBiases = np.zeros(shape=[self.no],dtype=np.float32)#output biases

        self.InitializeWeights()

    #compute the number of weights for all branches of the neural network
    @staticmethod
    def ComputeNumberOfWeights(ni,nh,no):
        numberOfWeights= (ni*nh)+(nh*no)+nh+no
        print("number of weights in network: ", numberOfWeights)
        return numberOfWeights

    def InitializeWeights(self):
        NumberOfWeights = self.ComputeNumberOfWeights(self.ni,self.nh,self.no)
        weights = np.zeros(shape=[NumberOfWeights],dtype=np.float32)
        range_min = -0.01
        range_max = 0.01
        for i in range(len(weights)):
            weights[i] = (range_max-range_min)*self.rnd.random()+range_min
        self.SetWeights(weights)

    def SetWeights(self,w):
        #loops convert the linear indexed weight vector in to 
        #matrix format for layers and vectors for biases

        idx = 0 
        #set the hidden layer weights
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i,j] = w[idx]
                idx += 1

        #set the hidden bias weights
        for i in range(self.nh):
            self.hBiases[i] = w[idx] 
            idx += 1

        #set the output layer weights
        for i in range(self.nh):
            for j in range(self.no):
                self.hoWeights[i,j] = w[idx]
                idx += 1

        #set the output bias weights
        for i in range(self.no):
            self.oBiases[i] = w[idx]
            idx += 1

    def GetWeights(self):
        NumberOfWeights = self.ComputeNumberOfWeights(self.ni,self.nh,self.no)
        result = np.zeros(shape=[NumberOfWeights],dtype=np.float32)
        idx = 0 
        
        #get the hidden layer weights
        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i,j] 
                idx += 1

        #get the hidden bias weights
        for i in range(self.nh):
            result[idx] = self.hBiases[i] 
            idx += 1

        #get the output layer weights
        for i in range(self.nh):
            for j in range(self.no):
                result[idx] = self.hoWeights[i,j]
                idx += 1

        #get the output bias weights
        for i in range(self.no):
            result[idx] = self.oBiases[i]
            idx += 1

        return result

#-------------------- Main Implementation-----------------------#

if __name__=="__main__":

    TrainX,TrainY,TestX,TestY = readMNIST()

    NumInputs = TrainX.shape[1] #give 784 inputs, one for each pixel in the images

    #Allow user inputs for number of hidden neurons
    while True:
        try:
            NumHidden = int(input("Input desired number of neurons in the hidden layer: "))

        except ValueError:
            print("ERROR: Number of Neurons must be an integer")

        else:
            break
    NumOutputs = np.max(TestY)+1 #Should always be 10 to represent digits 0-9
     
    print("number of classes: ",NumOutputs)
    
    print("The dimension of each training pattern: ",TrainX.shape[1])    # returns 784 ( = 28x28, in linear space)
    print("The number of training patterns: ",TrainX.shape[0])           # 55000 training patters
    print("The dimension of each test pattern: ",TestX.shape[1])         # 784
    print("The number of test patterns: ",TestX.shape[0])                #10,000

    print("The class labels of Training Patterns: ",TrainY[0:10])

    #since not using 1-hot gives vector of class human-readable class labels
    #i.e. if trainX is 7 then trainY is also 7
    print("The number of class labels of Training Patterns: ",TrainY.shape)
    print("The number of class labels of Test Patterns: ",TestY.shape) #number of test labels (10000)
    

    #PlotDigits(TrainX,10)

    nn = NeuralNetwork(NumInputs,NumHidden,NumOutputs,seed = np.random.randint(0,10))  

