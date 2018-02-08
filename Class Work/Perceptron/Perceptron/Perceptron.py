#Perceptron using Hebb's Rule

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


def PlotInput(data,labels):
    for d, sample in enumerate(data):#runs for the number of items in data
        if labels[d] == 0:#labels represents Y
            plt.scatter(sample[0],sample[1], s=120, marker = '_', linewidths = 2)#need to modify after reinstall to just plt.scatter
        else:
            plt.scatter(sample[0],sample[1], s=120, marker = '+', linewidths = 2)
    plt.plot([-2,6],[6,0.5])#surface line g(x) = 5.5x + 8y + 37 = 0
    plt.show()

#function to implement non-parametric perceptron ( or Hebb's Rule)
def perceptron(input,labels):
    w = np.zeros(len(input[0]))#creates weight vector of zeros of length of data patern (3 in this case) 
    w[0] = -5.5
    w[1] = -9.0
    w[2] = 37.0
    print("\nInitial weights: \n",w)
    learning_rate = 1
    NumberOfEpochs = 30 #max epochs 
    errors = []#empty vector to store errors after each iteration

    for iter in range(NumberOfEpochs): 
        total_error = 0
        yp = [] #predicted value of Y (desired class) 
        for i,x in enumerate(input): #move through each input training pattern
            v = np.dot(input[i],w)#represents local field w^T*X v is also known as net 
            if v > 0: #heaviside activation function (unipolar)
                yp.append(1)
            else:
                yp.append(0)
            total_error += abs(labels[i]-yp[i])#labels contains desired class values, (1 or 0) accumulates over the epoch
            w = w + learning_rate*(labels[i]-yp[i])*input[i] #update weight vector (notes 2/1 pg4)
        errors.append(total_error) #keeps track of error across epochs
        if total_error == 0: #(i.e. all patterns correctly classified in an epoch) 
            break #jumps out of outer loop
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()
    print("\nErrors: \n",errors)
    print("Epochs: ",len(errors))
    return w

def Evaluate(Weights, TestData, TestLabels,biasInput):
    for i,x in enumerate(TestData):#Classify testing data
        v = np.dot(TestData[i],Weights) # Field calcuation 
        if v > 0:
            TestLabels.append(1)
        else: 
            TestLabels.append(0)
        if TestLabels[i] == 0:
            plt.scatter(x[0],x[1],s=120,marker = '_', linewidth = 2)
        else:
            plt.scatter(x[0],x[1],s=120,marker = '+', linewidth = 2)
    slope = -Weights[0]/Weights[1] 
    intercept = -(biasInput*Weights[2])/Weights[1]

    axes  = plt.gca() #Get current axis
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept+slope*x_vals 
    plt.plot(x_vals,y_vals)
    plt.show()



#when running python program, this line prevents use in an imported module (i.e. cna't import this to another .py and use whats in __main__) 
if __name__ == '__main__': #how to designate a main in python
    #Define training data
    X = np.array([
        [-2,4],
        [4,1],
        [1,6],
        [2,4],
        [6,2],
        ])

    bias_input = -1 #plus or minus 1

    #augmented input (in class) 
    #X=np.array([
    #    [-2,4,bias_input],
    #    [4,1,bias_input],
    #    [1,6,bias_input],
    #    [2,4,bias_input],
    #    [6,2,bias_input],])


    #Loop augmentation of data with bias (more dynamically coded) 
    augment_data = np.zeros((X.shape[0],X.shape[1]+1)) #preallocate temp array to augment with bias value
    for i in range(len(X)):
        augment_data[i] = np.append(X[i],bias_input)
    X = augment_data

    #To get back original data
    #X = np.delete(X,X.shape[1]-1,1) #strips the bias off and returns original data array (data,index of bias,axis) 
    
    print("Input training pattern: ")
    print(X)

    #defining the desired class labels
    Y= np.array([0,0,1,1,1])#using unipolar discrete, first two in class one, rest in class 2
    PlotInput(X,Y)
    W = perceptron(X,Y)
    print("\nWeights: \n",W)

    #Testing 
    test_data = np.array([
        [-2,2,bias_input],
        [3,1,bias_input],
        [2,6,bias_input],
        [4,4,bias_input],
        [6,5,bias_input],])

    test_labels = [] #store classifications of test data 
    Evaluate(W,test_data,test_labels,bias_input)
    print("Test Data: \n",test_data)
    print("Test Labels:\n",test_labels)




