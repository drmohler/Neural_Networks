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
    
if __name__ == '__main__':
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














    




