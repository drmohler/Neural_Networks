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

#A Function to read data from csv files and create augemented patterns

def ReadImmunoData(filename,featureLabels):
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

def ReadCryoData(filename,featureLabels):
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
                arrx.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])]) #Read the actual data of the input features
                #if(type == "training"): #Read the desired output if the data is training data. (checks for string equality from params)
                arry.append(float(row[6]))
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
        TrainY[i] = np.asarray(labels[i])
        
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

if __name__ == '__main__':

    #boolean flags to choose data sets for training and testing
    ImmunoTrain = False  #True: Immuno training, False: Cryo training 
    ImmunoTest = True
    
    print("\nLoading training data ")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    tempfile = os.path.sep.join(current_dir.split(os.path.sep)[:-2])
    trainDataFile = os.path.join(tempfile,'Immunotherapy.csv')
    trainDataFileCryo = os.path.join(tempfile,'Cryotherapy.csv')
    labels = []
    labelsCryo = []
    trainDataMatrix,trainLabels = ReadImmunoData(trainDataFile,labels) #extract the immuno data
    trainDataCryo , trainLabelsCryo = ReadCryoData(trainDataFileCryo,labelsCryo) #extract cryo data
    
    #print("Data")
    
    #print(labels)
    #for i,x in enumerate(trainDataMatrix):
    #    print(trainDataMatrix[i])

    print("Total number of patterns: ", trainDataMatrix.shape[0])

    numTrain = int(trainDataMatrix.shape[0]*0.8) #use 80% of the data to train and the rest to test
    print("Number of patterns used for training: ", numTrain)

    #split the data set in to training and testing sets
    TrainX,TrainY,TestX,TestY = SplitData(trainDataMatrix,trainLabels,numTrain)
    TrainX_C,TrainY_C,TestX_C,TestY_C = SplitData(trainDataCryo,trainLabelsCryo,numTrain)

    #match the format of the labels to TF format for output
    TrainY = np.reshape(TrainY,(72,1))
    TestY  = np.reshape(TestY,(18,1))

    TrainY_C = np.reshape(TrainY_C,(72,1))
    TestY_C  = np.reshape(TestY_C,(18,1))
   
    
    #print("Training Patterns: ")
    #print("Labels:\n",labels)
    #print("TrainX:\n",TrainX)
    #print("TrainY:\n",TrainY) 

    #normalize training data
    normTrainX = NormalizeData(TrainX)
    normTrainX_C = NormalizeData(TrainX_C)

    #normailze testing data
    normTestX = NormalizeData(TestX) 
    normTestX_C = NormalizeData(TestX_C)
    
    #number of neurons in each layer


    if ImmunoTrain and ImmunoTest: #using only immuno data
        input = 6
        
    elif (ImmunoTrain) and (not ImmunoTest): #Train with immuno test w/ cryo
        normTrainX = np.delete(normTrainX,5,axis=1) #prune the data to match dimensions between Immuno and Cryo data
        input = 5
    elif (not ImmunoTrain) and (ImmunoTest): #Train with cryo test w/ immuno
        normTestX = np.delete(normTestX,5,axis=1) #prune the data to match dimensions between Immuno and Cryo data
        input = 5
    else:  #use only cryo data 
        input = 5   
    
    hidden = 25
    output = 1

    #Updating the weights, tf so it can be broadcasted
    #this code does BATCH UPDATE,  known as full batch
    
    num_epochs = 5000
    cost_value = []

    iNodes = tf.placeholder(tf.float32,[None,input])#input nodes, contains entire training data. 
                                                    #basically a matrix with 4 cols and rows are 
                                                    #number of training patterns 
    d_values = tf.placeholder(tf.float32,shape=(None,1)) #desired values (vector of 0 or 1) 

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

    learningRate = tf.constant(0.1)
    cost = tf.reduce_mean(tf.square(error)) #squared error cost function (no 1/2) 
    
    #select the optimizer for the network 
       #select the optimizer for the network 
    #step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    #step = tf.train.AdagradOptimizer(learningRate,0.1).minimize(cost)
    #step = tf.train.RMSPropOptimizer(learningRate,0.9).minimize(cost)
    step = tf.train.AdadeltaOptimizer(0.1).minimize(cost)
    #step = tf.train.MomentumOptimizer(0.1,0.2,False,'Momentum',True).minimize(cost)

    init = tf.global_variables_initializer() #fills all tf.variables

    if ImmunoTrain:
        trainingSet = normTrainX
        trainingLabels = TrainY
    else:
        trainingSet = normTrainX_C
        trainingLabels = TrainY_C
    if ImmunoTest:
        testSet = normTestX #test with immuno data
        testLabels = TestY
    else:
        testSet = normTestX_C #test with cryo data
        testLabels = TestY_C

    with tf.Session() as sess:
        sess.run(init)
        print("\n\nTraining...")
        for i in range(num_epochs): 
            sess.run(step,feed_dict={iNodes:trainingSet,d_values:trainingLabels})#execute step for however many patterns in X
            cost_value.append(sess.run(cost,feed_dict = {iNodes:trainingSet,d_values:trainingLabels}))
            error_price = sess.run(error,{iNodes:trainingSet,d_values:trainingLabels})

            if i%500 == 0:
                y_predTrain = sess.run(y,{iNodes:trainingSet}) #run a forward pass after training is complete (move inside loop if want every epoch
                print("Epoch: ",i,"\tMSE: ",cost_value[-1]) #display final cost value

        print("\nValidating Test Data ... \n")
        y_predTest = sess.run(y,{iNodes:testSet}) #pass the chosen testing set through the trained network 
    if ImmunoTrain and ImmunoTest: #using only immuno data
        print("Training and Testing: Immunotherapy Data")
    elif (ImmunoTrain) and (not ImmunoTest): #Train with immuno test w/ cryo
        print("Training: Immunotherapy |\tTesting Cryotherapy") 
    elif (not ImmunoTrain) and (ImmunoTest): #Train with cryo test w/ immuno
        print("Training: Cryotherapy |\tTesting Immunotherapy") 
    else:  #use only cryo data 
        print("Training and Testing: Cryotherapy Data")
    for i in range(trainingSet.shape[0]):
        if(y_predTrain[i]> 0.5):
            y_predTrain[i] = 1.0
        else:
            y_predTrain[i] = 0.0
    numMisClass = 0 
    for i in range(trainingSet.shape[0]):
        if(y_predTrain[i] != trainingLabels[i]):
            numMisClass += 1
            
    compare  = np.concatenate((trainingLabels,y_predTrain),axis=1)
    #print("Training Results:\n")
    #print("[True Network]")
    #print(compare)
    print("Total Training Misclassifications: ", numMisClass)

    for i in range(testSet.shape[0]):
        if(y_predTest[i]> 0.5):
            y_predTest[i] = 1.0
        else:
            y_predTest[i] = 0.0
    numMisClassTest = 0 
    for i in range(testSet.shape[0]):
        if(y_predTest[i] != testLabels[i]):
            numMisClassTest += 1
            
    compare  = np.concatenate((testLabels,y_predTest),axis=1)
    print("\nTraining Results:")
    print("[True Network]")
    print(compare)
    print("Total Validation Misclassifications: ", numMisClassTest)
    print("Network Accuracy: %", ((testSet.shape[0]-numMisClassTest)*100)/testSet.shape[0])

    #Cost vs Epoch Plot
    fig = plt.figure("Cost Vs Epoch for Cryo Network #1")
    plt.plot(cost_value)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title("Cost Vs Epoch for Cryo Network #1")
    plt.grid()
    #plt.show()