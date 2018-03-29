"""
David R Mohler 
EE-5410: Exam #1, Problem #2
03-29-2018

Back Propagation with 3 layers (2H 1O)
"""


#Module Imports
import random
import numpy as np 
import math
import sys

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
    def __init__(self,numInput,numHidden1,numHidden2,numOutput,seed):
        self.ni = numInput    #Input Dimension
        self.nh1 = numHidden1 #Number of 1st Hidden layer Neurons
        self.nh2 = numHidden2 #Number of 1st Hidden layer Neurons
        self.no = numOutput   #Number of Output Neuroms

        self.iNodes = np.zeros(shape=[self.ni],dtype=np.float32)   #input Neurons does not process.
        self.hNodes1 = np.zeros(shape=[self.nh1],dtype=np.float32)   #Hidden Neurons
        self.hNodes2 = np.zeros(shape=[self.nh2],dtype=np.float32)   #Hidden Neurons
        self.oNodes = np.zeros(shape=[self.no],dtype=np.float32)   #Output Neurons

        self.ihWeights = np.zeros(shape=[self.ni,self.nh1],dtype=np.float32)     #Hidden layer 1 Weight Matrix W1
        self.hhWeights = np.zeros(shape=[self.nh1,self.nh2],dtype=np.float32)     #Hidden layer 2 Weight Matrix W3
        self.hoWeights = np.zeros(shape=[self.nh2,self.no],dtype=np.float32)     #Output Layer Weight Matrix W3

        self.h1Biases = np.zeros(shape=[self.nh1],dtype=np.float32)           #Bias Weights of Hidden Layer 1
        self.h2Biases = np.zeros(shape=[self.nh2],dtype=np.float32)           #Bias Weights of Hidden Layer 2
        self.oBiases = np.zeros(shape=[self.no],dtype=np.float32)           #Bias Weights of Output Layer

        self.rnd = random.Random(seed)      #Execute Mulitple Instances

        self.InitializeWeights()

    #A Static Method to compute the total number of weights in the network
    @staticmethod
    def ComputeNumberOfWeights(numInputs,numHidden1, numHidden2 ,numOutput):
        wc = (numInputs*numHidden1) +(numHidden1*numHidden2) + (numHidden2*numOutput) + numHidden1 + numHidden2 + numOutput
        return wc

    #A Static Method to compute the activation output of a neuron
    @staticmethod
    def ActivationOutput(v): #implementation of sigmoid 
        out = 1.0/(1.0 + math.exp(-v))
        return out

    @staticmethod
    def ActivationOutputDeriv(v): #implementation of sigmoid 
        out = ActivationOutput(v)*(1.0-ActivationOutput(v))
        return out


    #Function to set the weights of different layers
    def SetWeights(self,w):
        idx = 0
        #Set the hidden layer weights
        for i in range(self.ni): #outer loop = # of nuerons in input layer 
            for j in range(self.nh1): #inner loop = # of nuerons in hidden layer 
                self.ihWeights[i,j] = w[idx]
                idx += 1

        #Set the hidden 1 bias weights
        for j in range(self.nh1):
            self.h1Biases[j] = w[idx]
            idx += 1


        for i in range(self.nh1): #outer loop = # of nuerons in input layer 
            for j in range(self.nh2): #inner loop = # of nuerons in hidden layer 
                self.hhWeights[i,j] = w[idx]
                idx += 1

        #Set the hidden 2 bias weights
        for j in range(self.nh2):
            self.h2Biases[j] = w[idx]
            idx += 1

        #set the ouptout layer weights
        for i in range(self.nh2):
            for j in range(self.no):
                self.hoWeights[i,j] = w[idx]
                idx += 1

        #Set the output bias weights
        for j in range(self.no):
            self.oBiases[j] = w[idx]
            idx += 1

    #Function to get the weights of different layers
    def GetWeights(self):
        NumberOfWeights = self.ComputeNumberOfWeights(self.ni,self.nh1,self.nh2,self.no)
        result = np.zeros(shape=[NumberOfWeights],dtype = np.float32)
        idx = 0

        for i in range(self.ni):
            for j in range(self.nh1):
                result[idx] = self.ihWeights[i,j]
                idx += 1
        
        for j in range(self.nh1):
            result[idx] = self.h1Biases[j]
            idx += 1


        for i in range(self.nh1):
            for j in range(self.nh2):
                result[idx] = self.hhWeights[i,j]
                idx += 1
        
        for j in range(self.nh2):
            result[idx] = self.h2Biases[j]
            idx += 1

        for i in range(self.nh2):
            for j in range(self.no):
                result[idx] = self.hoWeights[i,j]
                idx += 1

        for j in range(self.no):
            result[idx] = self.oBiases[j]
            idx += 1

        return result

    #A Method to intitialize the weights of the network.
    def InitializeWeights(self):
        NumberOfWeights = self.ComputeNumberOfWeights(self.ni,self.nh1,self.nh2,self.no)#total number of weights in entire network
        weights = np.zeros(shape=[NumberOfWeights],dtype=np.float32)
        range_min = -0.01
        range_max = 0.01
        for i in range(len(weights)): #one way of setting random values
            weights[i] = (range_max - range_min)*self.rnd.random() + range_min
        
        self.SetWeights(weights) #creates huge vector of weights with length ni*nh*no

    #A Method to perform forward pass
    def PerformForwardPass(self,xValues):
        h1Sums = np.zeros(shape=[self.nh1],dtype=np.float32)# a vector of local induced values (I1) 
        h2Sums = np.zeros(shape=[self.nh2],dtype=np.float32)# a vector of local induced values (I2) 
        oSums = np.zeros(shape=[self.no],dtype=np.float32)# I3

        #present the single input pattern to the network
        for i in range(self.ni):
            self.iNodes[i] = xValues[i]#iNodes: input neurons 
        #compute W1*X (not including bias here) --> gives you I1 
        for j in range(self.nh1):
            for i in range(self.ni):
                h1Sums[j] += self.iNodes[i]*self.ihWeights[i,j]#matrix vector multiplication 
        #Hidden Layer local induced field, now include the biases to get true I1
        for j in range(self.nh1):
            h1Sums[j] += self.h1Biases[j]

        #compute output of the hidden layer --> gives y1
        for j in range(self.nh1):
            self.hNodes1[j] = self.ActivationOutput(h1Sums[j])





                    #compute W1*X (not including bias here) --> gives you I1 
        for j in range(self.nh2):
            for i in range(self.nh1):
                h2Sums[j] += self.hNodes1[i]*self.hhWeights[i,j]#matrix vector multiplication 
        #Hidden Layer local induced field, now include the biases to get true I1
        for j in range(self.nh2):
            h2Sums[j] += self.h2Biases[j]

        #compute output of the hidden layer --> gives y1
        for j in range(self.nh2):
            self.hNodes2[j] = self.ActivationOutput(h2Sums[j])

        #compute W2*Y1
        for k in range(self.no):
            for j in range(self.nh2):
                oSums[k] += self.hNodes2[j]*self.hoWeights[j,k]
        
        #Output Layer local induced field, now include the biases to get true I2
        for j in range(self.no):
            oSums[j] += self.oBiases[j]

        #compute output of the output layer --> gives y2
        for j in range(self.no):
            self.oNodes[j] = self.ActivationOutput(oSums[j])

        return self.oNodes 

    #A Method to train the network
    def trainNN(self,TrainData,maxEpochs,learnRate):
        #do backprop and update weights in this function

        #component style,  compute gradient of each weight

        hoGrads = np.zeros(shape=[self.nh2,self.no],dtype=np.float32)
        obGrads = np.zeros(shape=[self.no],dtype=np.float32)#bias gradients

        hhGrads = np.zeros(shape=[self.nh1,self.nh2],dtype=np.float32)
        h2bGrads = np.zeros(shape=[self.nh2],dtype=np.float32)#bias gradients

        ihGrads = np.zeros(shape=[self.ni,self.nh1],dtype=np.float32)#gradients in to hidden layer
        h1bGrads = np.zeros(shape=[self.nh1],dtype=np.float32)#biases for neurons in hidden layer

        #output signals; gradients w/o associated input terms (y-d)*deriv of activation 
        oSignals = np.zeros(shape=[self.no],dtype=np.float32)

        #hidden signals; 
        h1Signals = np.zeros(shape=[self.nh1],dtype=np.float32)
        h2Signals = np.zeros(shape=[self.nh2],dtype=np.float32)

        x_values = np.zeros(shape=[self.ni],dtype=np.float32)   #A Single input pattern, row vector of 4
        d_values = np.zeros(shape=[self.no],dtype = np.float32) #The desired output, row vector of 3 

        numTrainItems = len(TrainData)#how many input patterns 
        print("Number of training patterns: ", numTrainItems)

        indices = np.arange(numTrainItems) # vector w/ 0,1,2,...,n-1

        epoch = 0
        while(epoch<MaxEpochs):
            self.rnd.shuffle(indices)#scramble the order of the vector
            for i in range(numTrainItems):
                idx = indices[i] #idx is the index of a training pattern, will be different in each epoch due to shuffling
                for j in range(self.ni):#get input values for particular training patterns
                    x_values[j] = TrainData[idx,j]
                for j in range(self.no): #get desired values for same pattern (this is done because they are combined, needs to be modified if sparate)
                    d_values[j] = TrainData[idx,self.ni+j]

                a_values = self.PerformForwardPass(x_values)

                #implement back propagation
                #compute the local gradient of the output layer
                for k in range(self.no):
                    O_derv = (1.0 - self.oNodes[k])*self.oNodes[k] #oNodes currently contains the output Y2
                    oSignals[k] = O_derv*(self.oNodes[k]-d_values[k])#calculates the gradient (y-d)*derv of activation

                #compute the hidden to output weight gradients using output local gradient and output of hidden units
                for j in range(self.nh2):
                    for k in range(self.no):
                        hoGrads[j,k] = oSignals[k]*self.hNodes2[j] #(y-d)*derv*Y1

                #compute the output node bias gradients
                for k in range(self.no):
                    obGrads[k] = oSignals[k] #bias contains 1 so just set equal to output signals



                #compute the local gradient of the hidden layer
                for j in range(self.nh2):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k]*self.hoWeights[j,k] 
                    h_derv = (1.0 - self.hNodes2[j])*self.hNodes2[j]#derivatives of hidden nodes
                    h2Signals[j] = sum*h_derv #contains W2^T*delta2


                """CONTINUE HERE WITH GRAD BETWEEN HIDDEN LAYERS"""

                #hidden weight gradients using hidden local gradient and input to the network
                for j in range(self.ni):
                    for k in range(self.nh):
                        ihGrads[j,k] = hSignals[k]*self.iNodes[j] # does schurs product and gives delta1

                #hidden node bias gradients
                for j in range(self.nh):
                    hbGrads[j] = hSignals[j]

                #here all error is back propagated, now update weights and biases using the gradients
                for j in range(self.ni):
                    for k in range(self.nh):
                        delta_wih = -1.0*learnRate*ihGrads[j,k]
                        self.ihWeights[j,k] += delta_wih
                #updating hidden node bias
                for j in range(self.nh):
                    delta_whb = -1.0*learnRate*hbGrads[j]
                    self.hBiases[j] += delta_whb
                
                for j in range(self.nh):
                    for k in range(self.no):
                        delta_who = -1.0*learnRate*hoGrads[j,k]
                        self.hoWeights[j,k] += delta_who

                for j in range(self.no):
                    delta_who = -1.0*learnRate*obGrads[j]
                    self.oBiases[j] += delta_who
            epoch += 1
            if epoch % 10 == 0:
                mse = self.ComputeMeanSquaredError(TrainData)
                print("Epcoh = ",epoch, "MSE = ",mse)

        weights = self.GetWeights()
        return weights



    def ComputeMeanSquaredError(self,data): #total of 120 training patterns, input dim =  4 output dim = 3
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni],dtype=np.float32)   #A Single input pattern, row vector of 4
        d_values = np.zeros(shape=[self.no],dtype = np.float32) #The desired output, row vector of 3 
        
        #loop to exctract values for each input pattern and then the corresponding desired output values 
        for i in range(len(data)):
            for j in range(self.ni):
                x_values[j] = data[i,j]     #Extract input values from the data row
            for j in range(self.no):
                d_values[j] = data[i,j+self.ni] #Extract desired values

            y_values = self.PerformForwardPass(x_values) #pass a single input pattern through and get the output

            for j in range(self.no):
                err = d_values[j] - y_values[j]
                sumSquaredError = err*err   #(d-y)^2

        return sumSquaredError/len(data)
    
    #Method to validate the network
    def Validate(self,data):
        numErrors = 0
        x_values = np.zeros(shape=[self.ni],dtype=np.float32)   #A Single input pattern
        d_values = np.zeros(shape=[self.no],dtype = np.float32) #The desired output
        Y = np.zeros(shape=[len(data),self.no],dtype=np.float32)

        for i in range(len(data)):
            for j in range(self.ni):
                x_values[j] = data[i,j]     #Extract input values from the data row
            for j in range(self.no):
                d_values[j] = data[i,j+self.ni] #Extract desired values

            y_values = (self.PerformForwardPass(x_values))
            for j in range(self.no):
                if(y_values[j] > 0.50):
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
            for j in range(self.no):
                if(abs(Y[i,j] - d_values[j]) == 1):
                    numErrors += 1
                    break
            print("y_values: ",y_values)
            print("d_values: ",d_values)

        return Y,numErrors

if __name__ == '__main__':
    print("\nLoading training data ")
    trainDataFile = "TrainData.txt"
    trainDataMatrix = loadFile(trainDataFile)
    print("Training Data")
    
    for i,x in enumerate(trainDataMatrix):
        print(trainDataMatrix[i])

    InputCount = 4      #Number of Inputs
    HiddenCount = 5     #Number of neurons in the hidden layer
    OutputCount = 3     #Number of Neurons in the output layer

    nn = NeuralNetwork(InputCount,HiddenCount,OutputCount,seed = 3)
    print("Hidden Layer Weights\n",nn.ihWeights)
    print("Hidden Layer Bias Weights\n",nn.hBiases)
    print("Output Layer Weights\n",nn.hoWeights)
    print("Output Layer Bias Weights\n",nn.oBiases)

    MaxEpochs = 100
    learnRate = 0.1
    print("Starting Training")
    nn.trainNN(trainDataMatrix,MaxEpochs,learnRate)

    Y,ErrorCount = nn.Validate(trainDataMatrix)
    print("Training Pattern Classification")
    print(Y)
    print("Number of Training Patterns = ",len(trainDataMatrix))
    print("Number of Training Patterns Misclassified = ",ErrorCount)
    input("Press any key to validate with test data")
    print("\nLoading Test data ")
    testDataFile = "TestData.txt"
    testDataMatrix = loadFile(testDataFile)
    print("Test Data")
    print("Number of Test Patterns = ",len(testDataMatrix))
    for i,x in enumerate(testDataMatrix):
        print(testDataMatrix[i])
    
    Y,ErrorCount = nn.Validate(testDataMatrix)
    print("Test Pattern Classification")
    print(Y)
    print("Number of Test Patterns = ",len(testDataMatrix))
    print("Number of Test Patterns Misclassified = ",ErrorCount)







