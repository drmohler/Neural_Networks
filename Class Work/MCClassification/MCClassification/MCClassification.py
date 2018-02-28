import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#Function to read the MNIST dataset along with the class labels
def readMNISTData():
    """read_data_sets returns a nested structure of python type objects to each 
    component of an element of the dataset"""
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)  
    """The category labels or variables are represented using the one-hot vector format, where the vector is all-zero
    apart from one element:
    Examples: Catergory 4: 00001000     Notice they are counted from the left. Will be 16 bit in tensorflow, only first ten are used
                Category 2: 00100000
                Category 0: 10000000"""

    # The MAGIC OF ONE HOT - these catagory labels will match the output of the softmax layer, with the probability of being in the 
    # right class at ~1, and the probability of the other classes being at ~0
    
    #breaking up data set into training and test data. A third set available is VALIDATION, but we're not using it yet
    train_X,train_Y,test_X,test_Y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
    return train_X,train_Y,test_X,test_Y

#Plotting the handwritten digits
# in the data base, each image is a 28x28 matrix of pixels
def PlotDigits(data,DigitCount):
    #subplots(nrows,ncols,
    f,a = plt.subplots(1,10,figsize = (10,2))
    for i in range(DigitCount):
        a[i].imshow(np.reshape(data[i],(28,28)))
    plt.show()

#Weight and Biases - 
def WeightsAndBiases(FeatureDim,ClassDim):
    X = tf.placeholder(tf.float32,[None,FeatureDim]) # placeholder for input, 'None' means we can hold whatever size the training or test data is
    Y = tf.placeholder(tf.float32,[None,ClassDim])   # same idea for output
    w = tf.Variable(tf.random_normal([FeatureDim,ClassDim],stddev=0.01),name='weights')
    b = tf.Variable(tf.random_normal([ClassDim]),name='bias_weights') # WTF W/ THE DIMENSION? Tensor flow only needs the dimension of the bias for the OUTPUT LAYER
    return X,Y,w,b

#Forward Pass
def ForwardPass(w,b,X):
    ypred = tf.matmul(X,w) + b 
    return ypred

#Cost function for softmax activation
def MultiClassCost(output,Y):
    #Multiclass Cost Function : C(theta) = -y*log(f(x,w))   where y is the true class
    # Harder way:   y = tf.matmul(X,w) + b   is known as unnormalized log probabilities because if we sum all those ys, it won't be equal to1
    # In tensorflow language, that's known as "logits". 
    # y_softmax = tf.nn.softmax(y) 
    # C = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_softmax),[1]))  I dont know
    # This method can cause NUMERICAL INSTABILITY and crash the program if implemented incorrectly

    # Here is the built in function that does everything in the comments above, logits are the unnormalized y from the forward pass func
    # To clarify, this function does both the feed forward portion of the network AND the softmax
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = Y))
    return cost

#Initialization

def Init():
    return tf.global_variables_initializer()

#Training Operation
def TrainingOp(LearningRate,mcCost):
    op_train = tf.train.GradientDescentOptimizer(LearningRate).minimize(mcCost)
    return op_train

if __name__ == '__main__':
    TrainX,TrainY,TestX,TestY = readMNISTData()
    print("The dimension of each training pattern: ",TrainX.shape[1])    # returns 784 ( = 28x28, in linear space)
    print("The number of training patterns: ",TrainX.shape[0])           # 55000 training patters
    print("The dimension of each test pattern: ",TestX.shape[1])         # 784
    print("The number of test patterns: ",TestX.shape[0])                #10,000

    print("The class labels of Training Patterns: ",TrainY[0:10])
    print("The number of class labels of Training Patterns: ",TrainY.shape)

    print("The number of class labels of Test Patterns: ",TestY.shape)
    PlotDigits(TrainX,10)

    X,Y,w,b = WeightsAndBiases(TrainX.shape[1],TrainY.shape[1])
    Y_Pred = ForwardPass(w,b,X)
    cost = MultiClassCost(Y_Pred,Y)

    learning_rate,epochs = 0.01,1000
    op_train = TrainingOp(learning_rate,cost)

    init = Init()
    cost_value = []
    accuracy_value = []

    #Execution 
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            # trains network with 55000 input patterns
            sess.run(op_train,feed_dict = {X:TrainX,Y:TrainY})
            # then calculates the cost function for each epoch
            loss_epoch = sess.run(cost,feed_dict = {X:TrainX, Y:TrainY})
            #argmax belonging to numpy. Returns the index value of the maximum value in the desired axis direction
            # example: pass [0,2,5,1,4,6] to argmax, it will return 5 because that's the index of 6
            # if you pass it a matrix without a dimension direction, it will convert the matrix to a flat array
            # if you pass a dimension, it wil search along that dimension and return a vector of either the max indexes
            # of all the columns, or all the rows
            
            accuracy_epoch = np.mean(np.argmax(sess.run(Y_Pred,feed_dict = {X:TrainX,Y:TrainY}),axis = 1) == np.argmax(TrainY,axis =1))
            # == np.argmax(TrainY,axis =1) returns index of actual class, checks if its equal to the predicted epoch.
            # this will be 0 if they're matched, and 1 if they're different, and then takes the mean from all the input
            # In htis case, a LOWER epoch means represents higher accuracey. 0 would be perfect. 

