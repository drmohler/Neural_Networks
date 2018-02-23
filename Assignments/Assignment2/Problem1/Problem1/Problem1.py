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
    maxEpochs = 2500 #Maximum allowable epochs
    cost =[] #List to store the cost value at each epoch (i.e. the classification errors) 

    for i in range(maxEpochs):
        EpochCost = 0
        ypred = [] #predicted output value
        for j,pattern in enumerate(AugInput):
            v = np.dot(AugInput[j],W)#Local field
            if v > 0:
                ypred.append(1)
            else:
                ypred.append(0)
            EpochCost += abs(labels[j]-ypred[j])
            W = W + learning_rate*(labels[j]-ypred[j])*AugInput[j]
        cost.append(EpochCost)
        if EpochCost == 0:
            print("Total epochs executed: ",str(i))
            break
    #return the trained network weight vector and the list containing the value of the cost function at the end of each epoch.
    return W,cost


#Function to determine the performance of the trained perceptron network.
def EvaluatePerfmon(Weights,TestData):
    ClassLabels = [] #A list to store the determined class lables.
    #Write code to determine the class labels of the test patterns and store the class labels in the ClassLabels list.
    for i,pattern in enumerate(TestData):
        v = np.dot(TestData[i],Weights)
        if v>0:
            ClassLabels.append(1)
        else:
            ClassLabels.append(0)
    #write code to return the ClassLabels as np.array type.
    ClassLabels = np.asarray(ClassLabels)#Use numpys built in function to convert to a numpy array
    print(type(ClassLabels)) #Prove that the list has been converted to np array
    return ClassLabels


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
    coeffs =np.array([Weights[0]/den, Weights[1]/den,-Weights[3]/den])
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

    #PlotData(X,Y,"Training Patterns")   #Plots the input patterns in 3D.

    #Reading test patterns
    XTest,YTest = ReadFileData(TestPatternFile,"",FeatureNames,bias_value)
    print("Augmented Test Patterns")
    print(XTest)
    print("Class Membership")
    print(YTest) #The YTest will be an emptynp.array since the program has to determine the class membership.

    #Comment the code below to examine the training and test patterns without completing the functions.
    W = GenerateRandomWeights(X)
    print("Initial Weights: ",W)
    W,C = PerceptronTrain(X,W,Y)
    print("Trained Weights: ",W)
    fig2 = plt.figure("Variation of Cost with Number of Epochs")
    plt.plot(C)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()

    YTest = EvaluatePerfmon(W,XTest)
    print("Class Membership: ", YTest)
    PlotData(XTest,YTest,"Test Patterns") 
    PlotDataWithSurface(X,XTest,Y,YTest,W) #custom function for plotting training and classified data together
    