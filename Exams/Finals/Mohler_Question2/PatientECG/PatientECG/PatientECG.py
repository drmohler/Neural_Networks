import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Function to read the .csv input data files and return the data as a numpy array
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

#Function to split the input patterns and class labels
def SplitData(data):
    x_values = np.zeros(shape = [len(data),len(data[0])-1],dtype = np.float32)
    d_values = np.zeros(shape = [len(data),1],dtype = np.float32)

    for i in range(len(data)):
        for j in range(len(data[0])-1):
            x_values[i,j] = data[i,j]
        d_values[i,0] = data[i,len(data[0])-1]

    return x_values,d_values

#Function to generated batches of training data of desired size
#size = batch size
#start is the starting index of a batch
def Batch(input,desired,size,start):
    x_values = np.zeros(shape = [size,len(input[0])],dtype = np.float32)
    d_values = np.zeros(shape = [size,1],dtype = np.float32)

    for i in range(size):
        for j in range(len(input[0])):
            x_values[i,j] = input[i+start,j]
        d_values[i,0] = desired[i+start,0]

    return x_values,d_values

#Function to determine the number of misclassifications.
def MisClassifications(yPred,yDesired):
    numMissClass = 0
    for i in range(len(yPred)):
        if(yPred[i,0] > 0.5):
            yPred[i,0] = 1.0
        else:
            yPred[i,0] = 0.0
        if(yPred[i,0] != yDesired[i,0]):
            numMissClass += 1
    return numMissClass

def NormalizeData(data):
    mean = np.mean(data,axis=0) #calculate the mean of the data down the columns  
    std = np.std(data,axis=0)   #calculate the std dev of the data down the cols
    norm = np.zeros(shape=[data.shape[0],data.shape[1]],dtype=np.float32)
    for j in range(data.shape[1]): #j should iterate through the input columns 
        for i in range(data.shape[0]): #iterate down a selected column 
            norm[i,j] = (data[i,j]-mean[j])/std[j]
    return norm

def sigmoid(v):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0),tf.exp(tf.negative(v))))
    
    
if __name__ == '__main__':

    #Load all data sets 
    ECGTrainFile = 'ECGTrainData.csv'
    ECGValFile = 'ECGValData.csv'
    ECGTestFile = 'ECGTestData.csv'

    print("\nLoading training data ")
    Data = loadFile(ECGTrainFile)
    trainX,trainY = SplitData(Data)
    print(trainX.shape)
    print(trainY.shape)

    print("\nLoading Validation data ")
    Data = loadFile(ECGValFile)
    ValX,ValY = SplitData(Data)
    print(ValX.shape)
    print(ValY.shape)

    print("\nLoading Test data ")
    Data = loadFile(ECGTestFile)
    TestX,TestY = SplitData(Data)
    print(TestX.shape)
    print(TestY.shape)

    #Normalize the data (may be unneccessary, test both ways) 
    trainX = NormalizeData(trainX)
    ValX = NormalizeData(ValX)
    TestX = NormalizeData(TestX) 

    num_epochs = 12000
    cost_value = [] #empty list to store cost at each epoch 

    
    inputs = trainX.shape[1]
    iNodes = tf.placeholder(tf.float32,[None,inputs])
    d_values = tf.placeholder(tf.float32,shape=(None,1)) #placeholder for test labels

    #set up necessary variables for dense network 

    output = 1 #single output neuron for binary classification
    
    hidden1 = 16 #number of neurons in first hidden layer
    w1 = tf.Variable(tf.truncated_normal([inputs,hidden1]))
    b1 = tf.Variable(tf.truncated_normal([1,hidden1]))

    hidden2 = 16
    w2 = tf.Variable(tf.truncated_normal([hidden1,hidden2]))
    b2 = tf.Variable(tf.truncated_normal([1,hidden2]))


    w3 = tf.Variable(tf.truncated_normal([hidden2,output]))
    b3 = tf.Variable(tf.truncated_normal([1,output]))

    #Implement the forward pass for the network
    i1 = tf.add(tf.matmul(iNodes,w1),b1) # WX + b 
    y1 = sigmoid(i1)

    i2 = tf.add(tf.matmul(y1,w2),b2)
    y2 = sigmoid(i2)

    i3 = tf.add(tf.matmul(y2,w3),b3)
    y = sigmoid(i3)

    #Implement Back Propagation
    error = tf.subtract(y,d_values)

    LR = tf.constant(1e-5)
    cost = tf.reduce_mean(tf.squared_difference(y,d_values))
    #cost = tf.reduce_mean(tf.square(error))

    #step = tf.train.GradientDescentOptimizer(LR).minimize(cost)
    step = tf.train.RMSPropOptimizer(LR).minimize(cost)
    #step = tf.train.AdamOptimizer(LR).minimize(cost)
    

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("\nTraining Network...")
        for i in range(num_epochs):
            sess.run(step,feed_dict={iNodes:trainX,d_values:trainY})
            cost_value.append(sess.run(cost,feed_dict={iNodes:trainX,d_values:trainY}))
            error_price = sess.run(error, {iNodes:trainX,d_values:trainY})
            y_predTrain = sess.run(y,{iNodes:trainX}) #pass train data through current model
            y_predVal = sess.run(y,{iNodes:ValX}) #pass val data through trained model

            TrainMiss = 0
            ValMiss = 0            
            TestMiss = 0 

            TrainMiss = MisClassifications(y_predTrain,trainY)
            ValMiss = MisClassifications(y_predVal,ValY)

            #for j in range(trainX.shape[1]):
            #    if y_predTrain[j]>0.5:
            #        y_predTrain[j] = 1.0
            #    else:
            #        y_predTrain[j] = 0.0

            #    if y_predTrain[j] != trainY[j]: 
            #        TrainMiss = TrainMiss + 1 

            #for j in range(ValX.shape[1]):
            #    if y_predVal[j]>0.5:
            #        y_predVal[j] = 1.0
            #    else:
            #        y_predVal[j] = 0.0

            #    if y_predVal[j] != ValY[j]: 
            #        ValMiss = ValMiss + 1 
            Train_Acc = ((trainX.shape[1]-TrainMiss)*100)/trainX.shape[1]
            Val_Acc = ((ValX.shape[1]-ValMiss)*100)/ValX.shape[1]

            if i%500 == 0:
                print("Epoch: ",i,"\tMSE: ",cost_value[-1],"\tTrain_Acc: %",
                      "{0:.2f}".format(Train_Acc),"\tVal_Acc: %","{0:.2f}".format(Val_Acc))
        y_predTest = sess.run(y,{iNodes:TestX}) # after last epoch run test data through model
      
    TestMiss = MisClassifications(y_predTest,TestY)    
    #for j in range(TestX.shape[1]):
    #            if y_predTest[j]>0.5:
    #                y_predTest[j] = 1.0
    #            else:
    #                y_predTest[j] = 0.0

    #            if y_predTest[j] != ValY[j]: 
    #                TestMiss = TestMiss + 1 

    print("\nTraining Misclassifications: ", TrainMiss)
    print("Validation Misclassifications: ", ValMiss)
    print("Test Misclassifications: ", TestMiss)
    print("\nNetwork Accuracy: %", ((TestX.shape[1]-TestMiss)*100)/TestX.shape[1])
    