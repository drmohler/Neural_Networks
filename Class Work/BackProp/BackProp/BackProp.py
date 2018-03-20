import random
import numpy as np
import math
import sys

#Method to read the input data file
def loadFile(df):
  # load a comma-delimited text file into an np matrix
    resultList = []
    f = open(df, 'r')
    for line in f:
        line = line.rstrip('\n')  
        sVals = line.split(',')   
        fVals = list(map(np.float32, sVals))  
        resultList.append(fVals) 
    f.close()
    return np.asarray(resultList, dtype=np.float32)
# end loadFile

class NeuralNetwork:
    def __init__(self,numInput,numHidden,numOutput,seed): #initialize network with given parameters
        self.ni = numInput  #Input Dimension
        self.nh = numHidden     #Number of Hidden layer Neurons
        self.no = numOutput     #Number of Output Neuroms

        self.iNodes = np.zeros(shape=[self.ni],dtype=np.float32)   #input Neurons does not process and passes the input values
        self.hNodes = np.zeros(shape=[self.nh],dtype=np.float32)   #Hidden Neurons
        self.oNodes = np.zeros(shape=[self.no],dtype=np.float32)   #Output Neurons

        self.ihWeights = np.zeros(shape=[self.ni,self.nh],dtype=np.float32)     #Hidden layer Weight Matrix W1
        self.hoWeights = np.zeros(shape=[self.nh,self.no],dtype=np.float32)     #Output Layer Weight Matrix W2

        self.hBiases = np.zeros(shape=[self.nh],dtype=np.float32)           #Bias Weights of Hidden Layer
        self.oBiases = np.zeros(shape=[self.no],dtype=np.float32)           #Bias Weights of Output Layer

        self.rnd = random.Random(seed)      #Execute Mulitple Instances

        self.InitializeWeights()

    #A Static Method to compute the total number of weights in the network
    @staticmethod
    def ComputeNumberOfWeights(numInputs,numHidden,numOutput):
        wc = (numInputs*numHidden) + (numHidden*numOutput) + numHidden + numOutput
        return wc

    #A Static Method to compute the activation output of a neuron
    @staticmethod
    def ActivationOutput(v):
        out = 1.0/(1.0 + math.exp(-v))  #sigmoid activation function
        return out


    #Function to set the weights of different layers
    def SetWeights(self,w):
        ...

    
    #A Method to intitialize the weights of the network.
    def InitializeWeights(self):
        ...

    #A Method to perform forward pass
    def PerformForwardPass(self,xValues):
        ...

    #A Method to train the network
    def trainNN(self,TrainData,maxEpochs,learnRate):
        ...


if __name__ == '__main__':
    print("\nLoading training and test data ")
    trainDataPath = "TrainData.txt"
    trainDataMatrix = loadFile(trainDataPath)
    print("Training Data")
    for i,x in enumerate(trainDataMatrix):
        print(trainDataMatrix[i])

    InputCount = 4      #Number of Inputs
    HiddenCount = 5     #Number of neurons in the hidden layer
    OutputCount = 3     #Number of Neurons in the output layer






