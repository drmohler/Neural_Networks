#Program to implement the delta rule using tanh activation function
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

tol = 0.01 #no longer binary error so need tolerance
#Function to implement the perceptron with gradient descent
def perceptron_delta(input,labels):
    W = np.zeros(len(input[0]))
    W = [1,-1,0,0.5]
    print("Initial Weights",W)

    learning_rate =0.1
    NumberOfEpochs = 500
    errors = []
    epoch_count = 0
    for  iter in range(NumberOfEpochs):
        epoch_count = iter+1
        total_error = 0
        y = []
        for i,x in enumerate(input): #number of patterns in each epoch
            v = np.dot(input[i],W) #local field
            y.append(math.tanh(v))#using hyperbolic tangent for -1-->1 range of values
            total_error += (labels[i]-y[i])*(labels[i]-y[i]) #(d-y)^2
            #uses stochastic gradient (i.e. the weights are updated with each training pattern) 
            W = W + learning_rate*(labels[i] - y[i])*(1.0 - (y[i]*y[i]))*input[i] #update weights (d-y)*tanhactderiv
            print("Epoch = {0:},Iteration = {1:},d = {2:},y = {3:.4f}, Error = {4:.6f}".format((iter+1),(i+1),labels[i],y[i],total_error))
        errors.append(total_error)
        if total_error < tol:
            break;
    print("Epoch = ",epoch_count,"\tFinal Error = ",total_error)
    return W, errors

if __name__ == '__main__':
    #Augmented Input Data
    X = np.array([
        [1,-2,0,-1],
        [0,1.5,-0.5,-1],
        [-1,1,0.5,-1],
        ])

    #Desired labels
    d = np.array([-1,-1,1])

    W, Errors = perceptron_delta(X,d)
    print("Final Weights W = ",W)
    plt.plot(Errors)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

