"""
David R Mohler
EE5410: Neural Networks
Assignment 4

Immunotherapy Network: 1
"""

import tensorflow
import tensorflow as tf
import numpy as np
import csv
from matplotlib import pyplot as plt
import os

"""MODIFY THIS TO READ THE IMMUNOTHERAPY DATA"""
# will need modifications that handle the data being split (arry)

#A Function to read data from csv files and create augemented patterns

def ReadFileData(filename,featureLabels):
    with open(filename,'rt') as f:  #Opening a file to read a text file
        reader = csv.reader(f)      #reader is a iteration object and each iteration returns a line in the file
        i = 0
        arrx = []
        arry = []
        for row in reader:
            if(i == 0): #Reading the first line which contains the names of the input features.
                featureLabels.append(row[1])    #First  Feature name
                featureLabels.append(row[2])    #Second Feature name
                featureLabels.append(row[3])    #Third  Feature name
                featureLabels.append(row[4])    #fourth Feature name
                featureLabels.append(row[5])    #fifth  Feature name
                featureLabels.append(row[6])    #fifth  Feature name
            else:
                arrx.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6])]) #Read the actual data of the input features
                #if(type == "training"): #Read the desired output if the data is training data. (checks for string equality from params)
                arry.append(float(row[7]))
            i = i + 1
        xInput = np.array(arrx) #Create an np.array object of the input features
        yInput = np.array(arry) #Create an np.array object of the desired output features. This will be empty for test patterns.

    return xInput,yInput
   

#A function used to split the given data in to training and test patterns
def SplitData(data,labels,NumTrain):
    TrainX = np.zeros(shape=[int(NumTrain),data.shape[1]],dtype=np.float32)
    TrainY = np.zeros(shape=[int(NumTrain)],dtype=np.float32)
    TestX = np.zeros(shape=[int(data.shape[0]-NumTrain),data.shape[1]],dtype=np.float32)
    TestY = np.zeros(shape=[int(data.shape[0]-NumTrain),1],dtype=np.float32)
    for i in range(NumTrain):
        TrainX[i] = data[i]
        TrainY[i] = labels[i]
        
    for i in range(int(data.shape[0]-NumTrain)):
            TestX[i] = data[i+NumTrain]
            TestY[i] = labels[i+NumTrain]
    
    return TrainX, TrainY, TestX, TestY

def NormalizeData(data):
    mean = np.mean(data,axis=0) #calculate the mean of the data down the columns (array 
    std = np.std(data,axis=0)   #calculate the std dev of the data down the cols
    norm = np.zeros(shape=[data.shape[0],data.shape[1]],dtype=np.float32)
    for j in range(data.shape[1]): #j should iterate through the input columns 
        for i in range(data.shape[0]): #iterate down a selected column 
            norm[i,j] = (data[i,j]-mean[j])/std[j]

    return norm

def sigmoid(v):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0),tf.exp(tf.negative(v))))
    #tf does broadcast which makes this very fast since it doesnt use looping, thats why it must all be tf variables

def sigmoid_prime(v):
    return tf.multiply(sigmoid(v),tf.subtract(tf.constant(1.0),sigmoid(v)))
    #derivative of the sigmoid function 

if __name__ == '__main__':
    print("\nLoading training data ")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    tempfile = os.path.sep.join(current_dir.split(os.path.sep)[:-2])
    trainDataFile = os.path.join(tempfile,'Immunotherapy.csv')
    labels = []
    trainDataMatrix,trainLabels = ReadFileData(trainDataFile,labels)
    #print("Data")
    
    #print(labels)
    #for i,x in enumerate(trainDataMatrix):
    #    print(trainDataMatrix[i])

    print("Total number of patterns: ", trainDataMatrix.shape[0])

    numTrain = int(trainDataMatrix.shape[0]*0.8)
    print("Number of patterns used for training: ", numTrain)

    #split the data set in to training and testing sets
    TrainX,TrainY,TestX,TestY = SplitData(trainDataMatrix,trainLabels,numTrain)
    
    print("Training Patterns: ")
    print("Labels:\n",labels)
    print("TrainX:\n",TrainX)
    print("TrainY:\n",TrainY) 

    #normalize training data
    normTrainX = NormalizeData(TrainX)
    
    #number of neurons in each layer
    input = 6
    hidden = 5
    output = 1

    #Updating the weights, tf so it can be broadcasted
    #this code does BATCH UPDATE,  known as full batch
    
    num_epochs = 10000
    cost_value = []

    iNodes = tf.placeholder(tf.float32,[None,input])#input nodes, contains entire training data. 
                                                    #basically a matrix with 4 cols and rows are 
                                                    #number of training patterns 
    d_values = tf.placeholder(tf.float32,[None]) #desired values (vector of 0 or 1) 

    #setting up variables 
    w1 = tf.Variable(tf.truncated_normal([input,hidden]))#use truncated normal to initialize weights (input x hidden) 
                                                         #cant combine bias in tf
    b1 = tf.Variable(tf.truncated_normal([1,hidden])) #for immunotherapy should be 1x6 array 
    w2 = tf.Variable(tf.truncated_normal([hidden,output])) #5x1
    b2 = tf.Variable(tf.truncated_normal([1,output])) #scalar value

    #the network architecture is created at this point

    #Implement forward pass
    i1 = tf.add(tf.matmul(iNodes,w1),b1)# WX+B this is also a broadcast operation
    y1 = sigmoid(i1) #push through activation function

    i2 = tf.add(tf.matmul(y1,w2),b2)
    y = sigmoid(i2)

    #Forward Pass Complete, now implement back propagation 
    error = tf.subtract(y,d_values)

    learningRate = tf.constant(0.05)
    cost = tf.reduce_mean(tf.square(error)) #squared error cost function (no 1/2) 
    
    #select the optimizer for the network 
    step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

    init = tf.global_variables_initializer() #fills all tf.variables

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_epochs):
            sess.run(step,feed_dict={iNodes:normTrainX,d_values:TrainY})#execute step for however many patterns in X
            cost_value.append(sess.run(cost,feed_dict = {iNodes:normTrainX,d_values:TrainY}))
            error_price = sess.run(error,{iNodes:normTrainX,d_values:TrainY})

            if i%1000 == 0:
                y_pred = sess.run(y,{iNodes:normTrainX}) #run a forward pass after training is complete (move inside loop if want every epoch
                print("Y_pred:",y_pred[0:10]) 
       
                print("Epoch: ",i,"\tMSE: ",cost_value[-1]) #display final cost value
    for i in range(normTrainX.shape[0]):
        if(y_pred[i]> 0.5):
            y_pred[i] = 1.0
        else:
            y_pred[i] = 0.0
    print("y_pred: ",y_pred[0:10],"\tTestY: ", TestY[0:10])
    numMisClass = 0 
    for i in range(normTrainX.shape[0]):
        if(y_pred[i] != TrainY[i]):
            numMisClass += 1
            break
    print("Total Misclassifications: ", numMisClass)