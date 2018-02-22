#Linear Regression with Tensor Flow

import tensorflow as tf
import numpy as np

#import boston data set from sklearn.datasets
from sklearn.datasets import load_boston 
import pandas as pd #Library for data analysis
from matplotlib import pyplot as plt 
import os

#Define a function to read the Boston housing data
def read_data():
    HouseData = load_boston()
    #The HouseData will be a dictionary variable. Keys are used to access dict data
    print(HouseData.keys())
    print(HouseData.data.shape)# (506,13) 506 training features, 13 features wide no bias or desired value
    print(HouseData.feature_names)
    print(HouseData.DESCR)#mean, stddev, and distribution statistics
    #Accessing the actual data
    hdata = pd.DataFrame(HouseData.data)
    hdata.columns = HouseData.feature_names #labels the columns
    hdata['price'] = HouseData.target #add column 14 to hold desired values
    print(hdata.head()) #shows large range of values (decimals to hunreds) can't directly use
    #Summary of the statistics
    print(hdata.describe())
    features = np.array(HouseData.data) #Assign features to numpy array
    target = np.array(HouseData.target)
    return features,target


def NormalizeFeatures(data):
    mean = np.mean(data,axis=0)#take the mean down the vertical axis (axis = 0)
    std = np.std(data,axis = 0)
    return (data-mean)/std

def append_bias(features,target):
    number_of_samples = features.shape[0] #Extract number of patterns (506) 
    number_of_features = features.shape[1] # "" width (13) 
    #create a vector of bias values
    bias_vector = np.ones((number_of_samples,1)) # can choose +or- 1, its arbitrary and gives same surface

    X = np.concatenate((features,bias_vector),axis=1) #add the vector to a new column at the end of patterns
    X = np.reshape(X,[number_of_samples,number_of_features+1])

    Y = np.reshape(target,[number_of_samples,1])
    return X,Y



if __name__ == '__main__':
    os.system('cls') #Get rid of dumb warning

    #Preprocessing of data 
    PricingFeatures, ActualPrice = read_data()
    NormalizedFeatures = NormalizeFeatures(PricingFeatures)
    print(NormalizedFeatures)
    X_input,Y_input = append_bias(NormalizedFeatures,ActualPrice)

    AugmentedFeatureCount = X_input.shape[1]
    print("Augmented Feature Size = : ", AugmentedFeatureCount)

    #creating the TF computational graph
    X = tf.placeholder(tf.float32,[None,AugmentedFeatureCount])  #'None' allows us to expand or contract number of patterns used
    Y = tf.placeholder(tf.float32,[None,1])  #output placeholder (output is price)
    w = tf.Variable(tf.random_normal((AugmentedFeatureCount,1)),name='weights')

    #Initialize the variables
    init = tf.global_variables_initializer()

    #Tensorflow ops and input params for placeholders, cost, and optimization
    learning_rate = 0.01
    num_epochs = 1000
    cost_value = [] #how cost changes at each epoch
    pred = tf.matmul(X,w) #Does prediction with Widrow-Hoff (i.e. no activation function) | will be 1D tensor/vector
    err = pred-Y 
    cost = tf.reduce_mean(tf.square(err)) #does not use 1/2 coeff from notes 

    #uses AutoGrad --> i.e. automatically finds gradient of cost function
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #Mean Square Error
    mse_summary = tf.summary.scalar('MSE',cost)#Required to gather the events for the tensorboard log
    writer = tf.summary.FileWriter("c:/tflogs_LR",tf.get_default_graph()) #writes graph without using session commands

    #Training
    with tf.Session() as sess: 
        sess.run(init) #run initialization of variables etc.
        for i in range(num_epochs):
            sess.run(train_op,feed_dict={X:X_input, Y:Y_input})#Begin training with known inputs and desired outputs
            cost_value.append(sess.run(cost,feed_dict = {X:X_input,Y:Y_input}))
            summary_str = mse_summary.eval(feed_dict = {X:X_input,Y:Y_input})#evaluate MSE
            writer.add_summary(summary_str,i) #write out MSE for each epoch
        #here the network is trained, now calculate the actual value of error and predicted price
        err_price = sess.run(err,{X:X_input,Y:Y_input})
        pred_price = sess.run(pred,{X:X_input,Y:Y_input})

    print("Mean Square Error: ", cost_value[-1])#[-1] is the last element in vector
    plt.plot(cost_value)
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    plt.show()

    plt.scatter(Y_input,pred_price)#If exactly correct, expect straight line
    plt.xlabel('Actual House Price')
    plt.ylabel('Prediceted House Price')
    plt.show()
    writer.close()

    '''NOTES: This is using the linear activation function. this is continuous and 
              gives us the monotonically decreasing cost. The scatter shows correlation
              b/w prediction and actual prices'''