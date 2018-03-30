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

#create a function to shape the data vector in to 28x28 matrices
def ReshapeData(patterns):
    ...

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


        self.rnd = random.Random(seed)

        self.InitializeWeights()

    @staticmethod
    def ActivationOutput(v): #implement the sigmoid activation function
        out = 1.0/(1.0 + math.exp(-v))
        return out

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

        print("ihweights dimensions: ",self.ihWeights.shape)
        

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

    def ComputeMSE(self,data,labels):
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni],dtype=np.float32) #a single input pattern (784)
        d_values = np.zeros(shape=[self.no],dtype=np.float32) # desired ouput(10) 


        indices = np.arange(len(data)) #vector with 0,1,2,...,n-1
        for i in range(len(data)):
            idx = indices[i]
            for j in range(self.ni):
                x_values[j] = data[i,j]
            for j in range(self.no):
                if labels[idx]==j:# look at desired integer output, in softmax want this prob to be near 1
                    d_values[j] = 1.0  
                else:
                    d_values[j] = 0.0

            y_values = self.ForwardPass(x_values)

            for j in range(self.no):
                err = d_values[j] - y_values[j]
                
                sumSquaredError += err**2
        return sumSquaredError/len(data)

    def ForwardPass(self,xValues):
        hsums = np.zeros(shape=[self.nh],dtype=np.float32) #store induced field vector I1 for hidden layer
        osums = np.zeros(shape=[self.no],dtype=np.float32) #store induced field vector I2 for output layer

        for i in range(self.ni):
            self.iNodes[i] = xValues[i] #assign pixel values from an input pattern to each input 
        for j in range(self.nh): #for each neuron in hidden layer perform W1*X = I1
            for i in range(self.ni):
                hsums[j] += self.iNodes[i]*self.ihWeights[i,j] #matrix vector multiplication

        for j in range(self.nh): #include the biases in the field vector
            hsums[j] += self.hBiases[j]

        #calculate the output of the neurons in the hidden layer and store in hNodes (Y1)
        for i in range(self.nh):
            self.hNodes[i] = self.ActivationOutput(hsums[i])
 
        #output layer local induced field (I2)
        for j in range(self.no):
            for i in range(self.nh):
                osums[j] += self.hNodes[i]*self.hoWeights[i,j]

        #include output layer biases
        for i in range(self.no):
            osums[i] +=  self.oBiases[i]

        #find output of the OL
        for i in range(self.no):
            self.oNodes[i] = self.ActivationOutput(osums[i])

        
        return self.oNodes #return the output of the network after a completed forward pass


    def trainNN(self,TrainData,TrainLabels,maxEpochs,learnRate):

        #preallocate all necessary vectors for gradients etc.

        hoGrads = np.zeros(shape=[self.nh,self.no],dtype=np.float32) #will be (# of hidden neurons) x 10 
        obGrads = np.zeros(shape=[self.no],dtype=np.float32) #contains gradient of the output biases
        ihGrads = np.zeros(shape=[self.ni,self.nh],dtype=np.float32) #will be  784 x (# of hidden neurons)
        hbGrads = np.zeros(shape=[self.nh],dtype=np.float32) #contains gradient of the hidden biases

        #output signals
        oSignals = np.zeros(shape=[self.no],dtype=np.float32)
        hSignals = np.zeros(shape=[self.nh],dtype=np.float32)

        x_values = np.zeros(shape=[self.ni],dtype=np.float32) #a single input pattern (784)
        d_values = np.zeros(shape=[self.no],dtype=np.float32) # desired ouput(10) 

        numTrainItems = len(TrainData) #total number of input training patterns
        print("Number of training patterns: ", numTrainItems)
        indices = np.arange(numTrainItems) #vector with 0,1,2,...,n-1

        #begin training 

        epoch = 0 
        errors = []
        while(epoch<maxEpochs):  
           # self.rnd.shuffle(indices) #scramble the order in which the patterns are presented to the network
            err_tot = 0
            for i in range(numTrainItems): #for all training patterns
                idx = indices[i]
                for j in range(self.ni): #POTENTIAL FOR ISSUES EXTRACTING DATA CORRECTLY HERE!!!
                    x_values[j] = TrainData[idx,j] #extract the values from a single input pattern (pixel values)
               
                for j in range(self.no):
                    if TrainLabels[idx]==j:# look at desired integer output, in softmax want this prob to be near 1
                        d_values[j] = 1.0  
                    else:
                        d_values[j] = 0.0  
                
                #if epoch == 0:
                #    print("X values for first pattern: ",x_values)
                #    print("shape of chosen pattern: ", x_values.shape) 
                    
                #    print("Desired outputs for first pattern: ", d_values)

                FP = self.ForwardPass(x_values) #while FP is not used later, the variables internal to nn are updated by forward pass

                #Back Propagate the error and update weights

                #local gradient of output layer
                for j in range(self.no):
                    O_deriv = self.oNodes[j]*(1.0-self.oNodes[j]) #derivative of the sigmaoid function at each output oNodes contains I2
                    oSignals[j] =  (self.oNodes[j]-d_values[j])*O_deriv #gives delta1 (y-d)*d/dw(I2) , I2 is the inputs to oNodes

                #hidden to output gradients
                for j in range(self.nh):
                    for k in range(self.no): 
                        hoGrads[j,k] = oSignals[k]*self.hNodes[j]#(y-d)*deriv*Y1

                for j in range(self.no):
                    obGrads[j] = oSignals[j] #bias = 1 so take signal as it is

                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k]*self.hoWeights[j,k]
                    h_deriv = self.hNodes[j]*(1.0-self.hNodes[j]) 
                    hSignals[j] = sum*h_deriv #W2^T * delta2

                #hidden node weight gradients
                for j in range(self.ni):
                    for k in range(self.nh):
                        ihGrads[j,k] = hSignals[k]*self.iNodes[j] #schurrs product to give delta1

                #hidden node bias gradients
                for j in range(self.nh):
                    hbGrads[j] = hSignals[j]

                #end back prop, now update weights
                for j in range(self.ni):
                    for k in range(self.nh):
                        delta_wih = -1.0*learnRate*ihGrads[j,k]
                        self.ihWeights[j,k] += delta_wih

                #updateing hidden node bias
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
            mse = self.ComputeMSE(TrainData,TrainLabels)
            errors.append(mse) #keep track of mse over epochs 
            print("Epoch: ", epoch, "MSE: ",mse)

        weights = self.GetWeights()
        print("errors: ", errors)
        plt.plot(errors)
        plt.show()
        return weights

    def validate(self,data,labels):
        numErrors = 0
        x_values = np.zeros(shape=[self.ni],dtype=np.float32) #a single input pattern (784)
        d_values = np.zeros(shape=[self.no],dtype=np.float32) # desired ouput(10) 
        Y = np.zeros(shape=[len(data),self.no],dtype = np.float32)

        indices = np.arange(len(data)) #vector with 0,1,2,...,n-1

        for i in range(len(data)):
            idx = indices[i]
            for j in range(self.ni):
                x_values[j] = data[i,j]
            for j in range(self.no):
                if labels[idx]==j:# look at desired integer output, in softmax want this prob to be near 1
                    d_values[j] = 1.0  
                else:
                    d_values[j] = 0.0

            y_values = self.ForwardPass(x_values)
            
            if i < 10:
                print("y_values: ", y_values)
                print("d values: ",d_values)  

            #POTENTIAL ISSUE IN ASSIGNING LABELS
            max_out = np.argmax(y_values)
            for j in range(self.no):
                if j == max_out:
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
            for j in range(self.no):
                if(abs(Y[i,j] - d_values[j]) == 1):
                    numErrors += 1
                    break
        return Y,numErrors
           
#------------------------------------------- Main Implementation--------------------------------------------#

if __name__=="__main__":

    learnRate = 0.75
    maxEpochs = 7

    TrainX,TrainY,TestX,TestY = readMNIST()

    NumInputs = TrainX.shape[1] #give 784 inputs, one for each pixel in the images

    ##Allow user inputs for number of hidden neurons
    #while True:
    #    try:
    #        NumHidden = int(input("Input desired number of neurons in the hidden layer: "))

    #    except ValueError:
    #        print("ERROR: Number of Neurons must be an integer")

    #    else:
    #        break
    NumHidden = 5
    NumOutputs = np.max(TestY)+1 #Should always be 10 to represent digits 0-9
     
    print("number of classes: ",NumOutputs)
    
    print("The dimension of each training pattern: ",TrainX.shape[1])    # returns 784 ( = 28x28, in linear space)
    print("The number of training patterns: ",TrainX.shape[0])           # 55000 training patters
    print("The dimension of each test pattern: ",TestX.shape[1])         # 784
    print("The number of test patterns: ",TestX.shape[0])                #10,000

    print("The class labels of Training Patterns: ",TrainY[0:10])

    #since not using 1-hot gives vector of class human-readable class labels
    #i.e. if trainX is 7 then trainY is also 7
    print("The number of class labels of Training Patterns: ",TrainY.size)
    print("The number of class labels of Test Patterns: ",TestY.size) #number of test labels (10000)
    
    print("Shape of input patterns", TrainX.shape)
    print("Shape of single pattern", TrainX[0].shape)


    nn = NeuralNetwork(NumInputs,NumHidden,NumOutputs,seed = np.random.randint(0,10))  

    print("Beginning Network Training")
    nn.trainNN(TrainX,TrainY,maxEpochs,learnRate)
    print("Network Training Complete\n") 
    print("Validating Results...\n")

    Y,ErrorCount = nn.validate(TrainX,TrainY) 
    print("First Ten Training Patterns:", TrainY[0:10]) 
    print("Training Pattern Classification")
    print(Y[0:10])
    print("Number of Training Patterns Misclassified = ",ErrorCount)

    print("\nValidating Testing Data...")
    Y,ErrorCount = nn.validate(TestX,TestY) 
    print("Number of Training Patterns Misclassified = ",ErrorCount)
    print("The Neural Network achieved %",((len(TestY)-ErrorCount)/len(TestY))*100.0," accuracy") 


