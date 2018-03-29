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
        for j in range(4): #four inputs and 3 outputs
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
    trainDataFile = "TrainData.txt"
    trainDataMatrix = loadFile(trainDataFile)
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

    localGradient_2 = tf.multiply(error,sigmoid_prime(i2)) #calculates delta 2 

    BiasGradient_2 = localGradient_2 #output layer biases

    temp = tf.matmul(localGradient_2,tf.transpose(w2))
    localGradient_1 = tf.multiply(temp,sigmoid_prime(i1))
    BiasGradient_1 = localGradient_1 #output layer biases

    #Compute the change in weights
    deltaW2 = tf.matmul(tf.transpose(y1),localGradient_2)
    deltaW1 = tf.matmul(tf.transpose(iNodes),localGradient_1)

    #Update the weights
   
    learningRate = tf.constant(0.05)
    step = [
        tf.assign(w1,tf.subtract(w1,tf.multiply(learningRate,deltaW1))), #update W1
        tf.assign(b1,tf.subtract(b1,tf.multiply(learningRate,tf.reduce_mean(BiasGradient_1,axis=[0])))),
        tf.assign(w2,tf.subtract(w2,tf.multiply(learningRate,deltaW2))),
        tf.assign(b2,tf.subtract(b2,tf.multiply(learningRate,tf.reduce_mean(BiasGradient_2,axis=[0]))))
        ]

    cost = tf.reduce_mean(tf.square(error)) #squared error cost function (no 1/2) 
    
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



   
            

