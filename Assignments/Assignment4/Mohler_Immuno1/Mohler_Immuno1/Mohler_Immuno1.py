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
from pathlib import Path
import os

"""MODIFY THIS TO READ THE IMMUNOTHERAPY DATA"""
# will need modifications that handle the data being split (arry)

#A Function to read data from csv files and create augemented patterns

def ReadFileData(filename,featureLabels,bias):
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
            else:
                arrx.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),bias]) #Read the actual data of the input features
                #if(type == "training"): #Read the desired output if the data is training data. (checks for string equality from params)
                arry.append(float(row[6]))
            i = i + 1
        xInput = np.array(arrx) #Create an np.array object of the input features
        yInput = np.array(arry) #Create an np.array object of the desired output features. This will be empty for test patterns.

    return xInput,yInput
   

def SplitData(data):
    x_values = np.zeros(shape = [len(data),4],dtype = np.float32)
    d_values = np.zeros(shape = [len(data),3],dtype = np.float32)

    for i in range(len(data)):
        for j in range(7): #four inputs and 3 outputs
            x_values[i,j] = data[i,j]
        for j in range(3):
            d_values[i,j] = data[i,4+j]

    return x_values,d_values

def sigmoid(v):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0),tf.exp(tf.negative(v))))
    #tf does broadcast which makes this very fast since it doesnt use looping, thats why it must all be tf variables

def sigmoid_prime(v):
    return tf.multiply(sigmoid(v),tf.subtract(tf.constant(1.0),sigmoid(v)))
    #derivative of the sigmoid function 

if __name__ == '__main__':
    print("\nLoading training data ")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    #print("Path: ", p)
    file = os.path.sep.join(current_dir.split(os.path.sep)[:-2])
    file = os.path.join(file,'\Immunotherapy.csv')
    print("file: ",file)

    trainDataFile = "Immunotherapy.csv"
    labels = []
    bias = -1
    trainDataMatrix = ReadFileData(trainDataFile,labels,bias)
    print("Training Data")
    
    for i,x in enumerate(trainDataMatrix):
        print(trainDataMatrix[i])

    X,d = SplitData(trainDataMatrix)
    print(X)
    print(d)

    #number of neurons in each layer
    input = 4
    hidden = 5
    output = 3

    #Updating the weights, tf so it can be broadcasted
    #this code does BATCH UPDATE,  known as full batch
    
    num_epochs = 5000
    cost_value = []

    iNodes = tf.placeholder(tf.float32,[None,input])#input nodes, contains entire training data. 
                                                    #basically a matrix with 4 cols and rows are 
                                                    #number of training patterns 
    d_values = tf.placeholder(tf.float32,[None,output]) #desired values 

    #setting up variables 
    w1 = tf.Variable(tf.truncated_normal([input,hidden]))#use truncated normal to initialize weights (input x hidden) 
                                                         #cant combine bias in tf
    b1 = tf.Variable(tf.truncated_normal([1,hidden])) 
    w2 = tf.Variable(tf.truncated_normal([hidden,output]))
    b2 = tf.Variable(tf.truncated_normal([1,output])) 

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
    
    #need 50000 epochs for equal performance to the 5000 by manual 
    step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost) #replaces the entire comment block ... gave wayyyy worse error 

    init = tf.global_variables_initializer()
    #fills all tf.variables, remember constants and variables are DIFFERENT

    #whatever you write in the session should be written above this for ease of use
    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_epochs):
            sess.run(step,feed_dict={iNodes:X,d_values:d})#execute step for however many patterns in X
            cost_value.append(sess.run(cost,feed_dict = {iNodes:X,d_values:d}))
            error_price = sess.run(error,{iNodes:X,d_values:d})

        y_pred = sess.run(y,{iNodes:X}) #run a forward pass after training is complete (move inside loop if want every epoch
    print("MSE: ",cost_value[-1]) #display final cost value
    for i in range(len(trainDataMatrix)):
        for j in range(3):
            if(y_pred[i,j]> 0.5):
                y_pred[i,j] = 1.0
            else:
                y_pred[i,j] = 0.0
    print("y_pred: ",y_pred)
    numMisClass = 0 
    for i in range(len(trainDataMatrix)):
        for j in range(3):
            if(y_pred[i,j] != d[i,j]):
                numMisClass += 1
                break
    print("Total Misclassifications: ", numMisClass)