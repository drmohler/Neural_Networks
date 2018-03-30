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
        self.h1Nodes = np.zeros(shape=[self.nh1],dtype=np.float32)   #Hidden Neurons
        self.h2Nodes = np.zeros(shape=[self.nh2],dtype=np.float32)   #Hidden Neurons
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
                h1Sums[j] += self.iNodes[i]*self.ihWeights[i,j]#matrix vector multiplication (X*W1)
        #Hidden Layer local induced field, now include the biases to get true I1
        for j in range(self.nh1):
            h1Sums[j] += self.h1Biases[j]

        #compute output of the hidden layer --> gives y1
        for j in range(self.nh1):
            self.h1Nodes[j] = self.ActivationOutput(h1Sums[j])

        #compute W2*Y1 (not including bias here) --> gives you I1 
        for j in range(self.nh2):
            for i in range(self.nh1):
                h2Sums[j] += self.h1Nodes[i]*self.hhWeights[i,j]#matrix vector multiplication 
        #Hidden Layer local induced field, now include the biases to get true I1
        for j in range(self.nh2):
            h2Sums[j] += self.h2Biases[j]

        #compute output of the hidden layer --> gives y2
        for j in range(self.nh2):
            self.h2Nodes[j] = self.ActivationOutput(h2Sums[j])

        #compute W3*Y2
        for k in range(self.no):
            for j in range(self.nh2):
                oSums[k] += self.h2Nodes[j]*self.hoWeights[j,k]
        
        #Output Layer local induced field, now include the biases to get true I2
        for j in range(self.no):
            oSums[j] += self.oBiases[j]

        #compute output of the output layer --> gives y2
        for j in range(self.no):
            self.oNodes[j] = self.ActivationOutput(oSums[j])

        return self.oNodes 


    """
    LOOK AT / REWRITE THE BACK PROP PORTION OF TRAINING,
    SPECIFICALLY NEED TO DERIVE THE GRADIENT FOR THE THIRD LAYER (MOST LIKELY
    ISSUES ARE OCCURING HERE!!!) 
    """

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
            #self.rnd.shuffle(indices)#scramble the order of the vector
            for i in range(numTrainItems):
                idx = indices[i] #idx is the index of a training pattern, will be different in each epoch due to shuffling
                for j in range(self.ni):#get input values for particular training patterns
                    x_values[j] = TrainData[idx,j]
                for j in range(self.no): #get desired values for same pattern (this is done because they are combined, needs to be modified if sparate)
                    d_values[j] = TrainData[idx,self.ni+j]
                if (epoch % 50 == 0) and i < 10:
                    print("xvalues",x_values)
                a_values = self.PerformForwardPass(x_values)
                if (epoch % 50 == 0) and i < 10:
                    print("after FP: ",np.round(a_values,2))
                    print("desired: ",d_values)
                    

                #implement back propagation
                #compute the local gradient of the output layer
                for k in range(self.no):
                    O_derv = (1.0 - self.oNodes[k])*self.oNodes[k] #oNodes currently contains the output Y3
                    oSignals[k] = O_derv*(self.oNodes[k]-d_values[k])#calculates the gradient (y-d)*derv of activation (Delta 3) 

                #compute the hidden to output weight gradients using output local gradient and output of hidden 2 units
                for j in range(self.nh2):
                    for k in range(self.no):
                        hoGrads[j,k] = oSignals[k]*self.h2Nodes[j] #delta3*Y2 

                #compute the output node bias gradients
                for k in range(self.no):
                    obGrads[k] = oSignals[k] #bias contains 1 so just set equal to output signals

                """ End Back prop of output layer """

                #compute the local gradient of the hidden layer
                for j in range(self.nh2):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k]*self.hoWeights[j,k] #del3*W3^T
                    h_derv = (1.0 - self.h2Nodes[j])*self.h2Nodes[j]#derivatives of hidden nodes phi'(I2)
                    h2Signals[j] = sum*h_derv #contains W3^T*delta2

                #hidden weight gradients using hidden local gradient and input to the network
                for j in range(self.nh1):
                    for k in range(self.nh2):
                        hhGrads[j,k] = h2Signals[k]*self.h1Nodes[j] # does schurs product and gives delta1

                #hidden layer 2 bias gradients
                for j in range(self.nh2):
                    h2bGrads[j] = h2Signals[j]

                """ End Back prop of Hidden layer 2 """

                #compute the local gradient of the hidden layer
                for j in range(self.nh1):
                    sum = 0.0
                    for k in range(self.nh2):
                        sum += h2Signals[k]*self.hhWeights[j,k] 
                    h_derv = (1.0 - self.h2Nodes[j])*self.h2Nodes[j]#derivatives of hidden nodes Y1
                    h1Signals[j] = sum*h_derv #contains W2^T*delta2*phi'(I2) = delta1

                 
                #hidden weight gradients using hidden local gradient and input to the network
                for j in range(self.ni):
                    for k in range(self.nh1):
                        ihGrads[j,k] = h1Signals[k]*self.iNodes[j] # does schurs product and gives delta1

                #hidden layer 2 bias gradients
                for j in range(self.nh1):
                    h1bGrads[j] = h1Signals[j]

                """ End Back prop of Hidden layer 2 (POTNENTIAL ERROR, MIGHT NEED MORE TERMS FOR GRADIENT)"""


                #here all error is back propagated, now update weights and biases using the gradients
                for j in range(self.ni):
                    for k in range(self.nh1):
                        delta_wih = -1.0*learnRate*ihGrads[j,k]
                        self.ihWeights[j,k] += delta_wih
                #updating hidden node bias
                for j in range(self.nh1):
                    delta_wh1b = -1.0*learnRate*h1bGrads[j]
                    self.h1Biases[j] += delta_wh1b
                
                for j in range(self.nh1):
                    for k in range(self.nh2):
                        delta_whh = -1.0*learnRate*hhGrads[j,k]
                        self.hhWeights[j,k] += delta_whh
                #updating hidden node bias
                for j in range(self.nh2):
                    delta_wh2b = -1.0*learnRate*h2bGrads[j]
                    self.h2Biases[j] += delta_wh2b


                for j in range(self.nh2):
                    for k in range(self.no):
                        delta_who = -1.0*learnRate*hoGrads[j,k]
                        self.hoWeights[j,k] += delta_who
                for j in range(self.no):
                    delta_who = -1.0*learnRate*obGrads[j]
                    self.oBiases[j] += delta_who


            epoch += 1
            if epoch % 50 == 0:
                mse = self.ComputeMeanSquaredError(TrainData)
                print("Epoch = ",epoch, "MSE = ",mse)
                print("Hidden Layer 1 Weights\n",self.ihWeights)
                print("Hidden Layer 1 Bias Weights\n",self.h1Biases)
                print("Hidden Layer 2 Weights\n",self.hhWeights)
                print("Hidden Layer 2 Bias Weights\n",self.h2Biases)
                print("Output Layer Weights\n",self.hoWeights)
                print("Output Layer Bias Weights\n",self.oBiases)
                

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
                sumSquaredError += err*err   #(d-y)^2

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
    HiddenCount1 = 5     #Number of neurons in the hidden layer
    HiddenCount2 = 5     #Number of neurons in the hidden layer
    OutputCount = 3     #Number of Neurons in the output layer

    nn = NeuralNetwork(InputCount,HiddenCount1,HiddenCount1,OutputCount,seed =  np.random.randint(0,10))
    print("Hidden Layer 1 Weights\n",nn.ihWeights)
    print("Hidden Layer 1 Bias Weights\n",nn.h1Biases)
    print("Hidden Layer 2 Weights\n",nn.hhWeights)
    print("Hidden Layer 2 Bias Weights\n",nn.h2Biases)
    print("Output Layer Weights\n",nn.hoWeights)
    print("Output Layer Bias Weights\n",nn.oBiases)

    MaxEpochs = 250
    learnRate = 0.8
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







