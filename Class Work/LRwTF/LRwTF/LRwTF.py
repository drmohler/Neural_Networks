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
    #Now ready to do computational graph
