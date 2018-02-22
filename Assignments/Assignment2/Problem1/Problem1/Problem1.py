#Problem 1
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

#Function to initialize the weight vector with random values between 0.0 and 1.0
def GenerateRandomWeights(AugInput):
    NumberOfFeatures = len(AugInput[0]) - 1
    print("Number of Features = ",NumberOfFeatures)
    weights = np.zeros(NumberOfFeatures+1)
    weights[0:NumberOfFeatures] = np.random.random_sample([1,NumberOfFeatures])
    return weights
    """Write code to initialize the weight vector of NumberOfFeatures elements with random values
   in the range 0.0 to 1.0. You need to use the np.random.ramdom_sample function to generate numbers
   between 0.0 and 1.0. The weight corresponding to the bias should be initialized to zero."""

   #Write code to return the initialized weight vector

#Function to train the perceptron
def PerceptronTrain(AugInput,W,labels):
    learning_rate = 1.0
    #Write code to train the network and collect the value of the cost function at the end of each epoch.



    #return the trained network weight vector and the list containing the value of the cost function at the end of each epoch.


#Function to determine the performance of the trained perceptron network.
def EvaluatePerfmon(Weights,TestData):
    ClassLabels = [] #A list to store the determined class lables.
    #Write code to determine the class labels of the test patterns and store the class labels in the ClassLabels list.


    #write code to return the ClassLabels as np.array type.


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

    #Reading test patterns
    XTest,YTest = ReadFileData(TestPatternFile,"",FeatureNames,bias_value)
    print("Augmented Test Patterns")
    print(XTest)
    print("Class Membership")
    print(YTest) #The YTest will be an emptynp.array since the program has to determine the class membership.

    #Comment the code below to examine the training and test patterns without completing the functions.
    W = GenerateRandomWeights(X)
    print("W: ",W)
    #W,C = PerceptronTrain(X,W,Y)
    #print(W)
    #fig2 = plt.figure("Variation of Cost with Number of Epochs")
    #plt.plot(C)
    #plt.xlabel('Epoch')
    #plt.ylabel('Cost')
    #plt.show()

    #YTest = EvaluatePerfmon(W,XTest)
    #print(YTest)