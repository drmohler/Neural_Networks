"""
David Mohler
EE-5410: Neural Networks
Assignment #3: Problem 3
Continous perceptron implementation
"""
import math #use for exponential in sigmoid
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #Library for 3D plotting
import csv  #Python library to read excel csv files

#A Function to read data from csv files and create augemented patterns
def ReadFileData(filename,type,featureLabels,bias):
    with open(filename,'rt') as f:  #Opening a file to read a text file
        reader = csv.reader(f)      #reader is a iteration object and each iteration returns a line in the file
        i = 0
        arrx = []
        arry = []
        for row in reader:
            if(i == 0): #Reading the first line which contains the names of the input features.
                featureLabels.append(row[1])    #First Feature name
                featureLabels.append(row[2])    #Second Feature name
                featureLabels.append(row[3])    #Third Feature name
            else:
                arrx.append([float(row[1]),float(row[2]),float(row[3]),bias]) #Read the actual data of the input features
                if(type == "training"): #Read the desired output if the data is training data. (checks for string equality from params)
                    arry.append(float(row[4]))
            i = i + 1
        xInput = np.array(arrx) #Create an np.array object of the input features
        yInput = np.array(arry) #Create an np.array object of the desired output features. This will be empty for test patterns.

    return xInput,yInput

#Function to plot data
def PlotData(Features,Labels,Title):
    fig = plt.figure(Title)     #Create a figure object
    ax = fig.add_subplot(111, projection='3d')  #Create a subplot with 3D projection
    for d, sample in enumerate(Features):
        if Labels[d] == 0:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'red', marker = "_",linewidth = 2)
        else:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'blue', marker = "+",linewidth = 2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()


#Function to plot the training data and the classified features, along with the decision hypersurface
def PlotDataWithSurface(TrainingFeatures,TestFeatures,TrainingLabels,TestLabels,Weights):    
    fig = plt.figure('Classified Test Features and Training Data')     #Create a figure object
    ax = fig.add_subplot(111, projection='3d')  #Create a subplot with 3D projection
    
    #Plot training data 
    for d, sample in enumerate(TrainingFeatures):
        if TrainingLabels[d] == 0:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'red', marker = "_",linewidth = 2)
        else:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'blue', marker = "+",linewidth = 2)

    #Plot classified test features
    for t, sample in enumerate(TestFeatures):
        if TestLabels[t] == 0:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'green', marker = "_",linewidth = 2)
        else:
            ax.scatter(sample[0],sample[1],sample[2], s = 120, color = 'cyan', marker = "+",linewidth = 2)

    #Generate planar equation from trained weights
    den = -Weights[2]
    coeffs =np.array([Weights[0]/den, Weights[1]/den,-Weights[3]/den]) #extract the coefficients of the equation of the format Z = ax+by+c
    print('Equation: Z=', str(coeffs[0]),'X +',str(coeffs[1]),'Y +',str(coeffs[2])) 
     
    vals = np.linspace(-2,2)
    X,Y = np.meshgrid(vals,vals)
    Z = np.zeros(np.shape(X))
    
    for i in range(len(X)):
        Z[i] = coeffs[0]*X[i]+coeffs[1]*Y[i]+coeffs[2] #Equation of the hypersurface (plane) 
    
    ax.plot_surface(X,Y,Z)
    ax.view_init(elev=1,azim=-31)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()

#Function to initialize the weight vector with random values between 0.0 and 1.0
def GenerateRandomWeights(AugInput):
    NumberOfFeatures = len(AugInput[0]) - 1
    print("Number of Features = ",NumberOfFeatures)
    weights = np.zeros(NumberOfFeatures+1)
    weights[0:NumberOfFeatures] = np.random.random_sample([1,NumberOfFeatures])
    return weights

#Short function to implement the sigmoid activation function
def sigmoid(v):
    y = 1/(1+math.exp(-v))
    return y

#function used to implement the generalized delta rule for the continuous perceptron
def perceptron_delta(input,W,labels):
    tol = 0.1 #tallowable tolerance for error in continuous perceptron 
    maxEpochs = 1000000 #Large number of epochs for case with low tolerance and small learning rate 
    learning_rate = 0.25
    errors = []
    epoch_count = 0

    for iter in range(maxEpochs):
        epoch_count = iter+1
        total_error = 0 
        y =[]
        for i,x in enumerate(input):
            v = np.dot(input[i],W) #calculate induced local field (Net) 
            y.append(sigmoid(v)) #use sigmoid activation to obtain classifications between 0 and 1
            total_error += (labels[i]-y[i])*(labels[i]-y[i]) #(d-y)^2 
            #apply stochastic gradient to update weights with each training pattern
            W = W + learning_rate*(labels[i]-y[i])*(y[i]*(1-y[i]))*input[i] #update the weights 
            #uncomment to see print of every 1000th epoch and its current error 
            #if (iter%1000 == 0) or ((total_error<tol)and(i==29)):
            #    print("Epoch = {0:},Iteration = {1:},d = {2:},y = {3:.4f}, Error = {4:.6f}".format((iter+1),(i+1),labels[i],y[i],total_error))
        errors.append(total_error)
        if total_error < tol:
            break
    print("Epoch: ", epoch_count, "\tFinal Error: ",total_error)
    #return the weights and the sum of the squared errors (cost) 
    return W,errors

def evaluate(weights,testData):
    ClassLabels = [] 
    for i,x in enumerate(testData):
        v = np.dot(testData[i],weights) #calculate induced local field (Net) 
        ClassLabels.append(int(np.round(sigmoid(v)))) #use sigmoid activation to obtain classifications between 0 and 1, round the value to obtain class label
    ClassLabels = np.asarray(ClassLabels)
    return ClassLabels #return the classifications made by the network to the user 


#---------------------------------------------------------------MAIN IMPLEMENTATION ----------------------------------------------------------------------------#
if __name__ == '__main__':
    bias_value = -1
    #Reading Training Data
    TrainingPatternFile = "TrainingPatterns.csv"
    TestPatternFile = "TestPatterns.csv"
    FeatureNames = [] #A vector to store the input training feature names
    X,Y = ReadFileData(TrainingPatternFile,"training",FeatureNames,bias_value)   #X and Y  will be of type np.array
    print(FeatureNames) #Displays the feature names
    print("Augmented Training Patterns")
    print(X)    #Prints the input feature values
    print("Desired Class Membership")
    print(Y)    #Print the desired output of input pattern

    PlotData(X,Y,"Training Patterns")   #Plots the input patterns in 3D.

    XTest,YTest = ReadFileData(TestPatternFile,"",FeatureNames,bias_value)
    print("Augmented Test Patterns")
    print(XTest)
    print("Class Membership")
    print(YTest) #The YTest will be an emptynp.array since the program has to determine the class membership.
    W = GenerateRandomWeights(X) #generate initial weights between 0 and 1 for x1 x2 and x3 
    print("Initial Weights: ",W)

    #Perform training of the network using delta-training rule
    W,C = perceptron_delta(X,W,Y) #pass the training patterns, random weights, and desired classification to the network
    print("Trained Weights: ",W)

 
    #Pass the test patterns through the trained network and classify them 
    YTest = evaluate(W,XTest)
    print("Class Membership: ", YTest)

    fig2 = plt.figure("Variation of Cost with Number of Epochs")
    plt.plot(C)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()

    PlotDataWithSurface(X,XTest,Y,YTest,W) #display classified data with the given test patterns and hyper surface