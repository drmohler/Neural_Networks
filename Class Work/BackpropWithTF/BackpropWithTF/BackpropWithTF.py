import tensorflow
import tensorflow as tf
import numpy as np

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

def SplitData(data):
    x_values = np.zeros(shape = [len(data),4],dtype = np.float32)
    d_values = np.zeros(shape = [len(data),3],dtype = np.float32)

    for i in range(len(data)):
        for j in range(4):
            x_values[i,j] = data[i,j]
        for j in range(3):
            d_values[i,j] = data[i,4+j]

    return x_values,d_values

def sigmoid(v):
    ...
    

def sigmoid_prime(v):
    ...

if __name__ == '__main__':
    print("\nLoading training data ")
    trainDataFile = "TrainData.txt"
    trainDataMatrix = loadFile(trainDataFile)
    print("Training Data")
    
    for i,x in enumerate(trainDataMatrix):
        print(trainDataMatrix[i])

    X,d = SplitData(trainDataMatrix)
    print(X)
    print(d)

    input = 4
    hidden = 5
    output = 3

    

    #Updating the weights
    learningRate = tf.constant(0.05)

    
    num_epochs = 500
    cost_value = []

   
            

