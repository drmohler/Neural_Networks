import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

"""CNN accepts images of height 28, width 28, and depth 1 corresponding to gray scale.
The images are applied to the first convolution layer and produces 64 feature maps.
(64 kernels for the first layer) 

The second convolution layer provides 128 feature maps.
The feature maps are 2D.
The fully connected layers has 1024 units.
The max pooling layers are designed t reduce the feature map size by 1/4."""

#Function to read the MNIST dataset along with the class labels
def readMNISTData():
    """read_data_sets returns a nested structure of python type objects to each 
    component of an element of the dataset"""
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)  
    return mnist

#Function to create Convolution Layer
def conv2d(X,W,b,strides=1):
    """
    The tf.nn.conv2d executes the tensorflow convolution operation.
    Four stride parameters are provided for the four dimensions of the tensor.
    The first and the last stride parameters have to be one since the first dimension 
    represents the image index and the last dimension represents the channel index.
    The second and third stride values performing striding on the image height and width.
    """
    """
    If the padding = 'SAME', the input and output images are of the same size by implementing
    zero padding on the input. (TF will compute using the padding equation from notes 4-12-2018) 
    If the padding = 'VALID', the input is not padded and the output image size will be less 
    than the input image.
    """
    net = tf.nn.conv2d(X,W,strides=[1,strides,strides,1],padding='SAME')
    net = tf.nn.bias_add(net,b) #add bias to each convolved value, but all get the same bias value
    return tf.nn.relu(net) #return the output of the detection layer

#Function to create Pooling
def maxPool2d(X,stride=2):
    """
    The maximum pooling is performed on the locality of 2x2 which is described by the ksize.
    The pooling is performed by moving two pixels both in height and width dimensions.
    """
    #compresses data size by a factor of 4
    #k size --> input to max pooling 
    return tf.nn.max_pool(X,ksize = [1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')

#Function to Create the Feed Forward Network
def conv_net(X,weights,biases,dropout):


    #Reshape the input 1D image into 4 dimensional tensor
    
    X = tf.reshape(X,shape=[-1,28,28,1]) #X is whole batch, reshapes each of them
    #Convolution Layer 1
    conv1 = conv2d(X,weights['wc1'],biases['bc1'])#wc1 --> weights to conv layer 1
    conv1 = maxPool2d(conv1,2)
    #Convolution Layer 2
    
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])#wc1 --> weights to conv layer 1
    conv2 = maxPool2d(conv2,2)

    #Fully Connected Layer Operation
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]]) #wd --> dense layer 
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    

    #Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)#dropout is probability of dropout
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

    #Output Class Prediction
    ...

if __name__ == '__main__':
    mnist = readMNISTData()
    print("The dimension of each training pattern: ",mnist.train.images.shape[1])
    print("The number of training patterns: ",mnist.train.images.shape[0])
    print("The dimension of each test pattern: ",mnist.test.images.shape[1])
    print("The number of test patterns: ",mnist.test.images.shape[0])

    print("The class labels of Training Patterns: ",mnist.train.labels[0:10])
    print("The number of class labels of Training Patterns: ",mnist.train.labels.shape)

    print("The number of class labels of Test Patterns: ",mnist.test.labels.shape)

    learning_rate = 0.01
    epochs = 20
    batch_size = 256
    num_batches = int(mnist.train.num_examples/batch_size)
    print("Number of Batches = ",num_batches)

    n_classes = mnist.train.labels.shape[1]
    print("Number of Classes = ",n_classes)

	#MNIST data size (28x28 BW images) 
    img_height = 28
    img_width = 28

    #Keep Probability Definition
    keep_prob = tf.placeholder(tf.float32)
    dropout = 0.75 #probability of dropout

    #Kernel Definition (5x5 convolution kernel) 
    filter_height = 5
    filter_width = 5

    #Depth Definition
    depth_in = 1 #BW so only one channel
    depth_out1 = 64 #output of firts kernel will have depth of 64, so 64 feature maps
    depth_out2 = 128 #outputs 128 feature maps

    #Fully Connected Layer Definition
    FCL_Neurons = 1024 #1024 neurons 

    #Input Output Definition
    x = tf.placeholder(tf.float32,[None,mnist.train.images.shape[1]])
    """This is equivalent to writing:
       x = tf.placeholder(tf.float32,[None,img_height,img_width,1])
    """
    y = tf.placeholder(tf.float32,[None,mnist.train.labels.shape[1]])

    display_step = 1

    #Weight Definitions of Convolution layers
    """
    Number of Neurons in conv1: img_height*img_width*depth_out1 (28x28x64)
    Number of Weights of Convolution Layer 1 with padding
    wc1: filter_height*filter_width*depth_in*depth_out1
    wc1 is a tensor of dimension 5x5x1x64 (kernel dim x 1 channel x 64 maps) 
    """

    """
    Number of Neurons in conv2: img_height*img_width*depth_out2 (28x28x128)
    Number of Weights of Convolution Layer 2 with padding
    wc2: filter_height*filter_width*depth_out1*depth_out2
    wc2 is a tensor of dimension 5x5x64x128 
	(kernel dim x 64 channels x 128 output maps) 
    """

    """Each Max pooling reduces the image size by 1/4 and in total by 1/16
    There are depth_out2 feature maps of 1/16th size of the original image.
    The depth_out2 feature maps is the input to the fully connected layer with
    total number of connections= (1/16)*image_height*image_width*depth_out2*FCL_Neurons"""


    """The conv1 has 64 feature maps resulting in 64 biases.
    The conv2 has 128 feature maps resulting in 128 biases.
    The fully connected layer has 1024 neurons resulting in 1024 biases
    The output layer contains n_classes neurons resulting in n_classes bias"""

    
    weights = {
        'wc1' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])), #(5x5x1x64) 
        'wc2' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])), #(5x5x64x128) if not using same padding the first parameters will change
        'wd1' : tf.Variable(tf.random_normal([7*7*depth_out2,FCL_Neurons])), #flatten the data for FC layer
        'out' : tf.Variable(tf.random_normal([FCL_Neurons,n_classes])) #(1024x10)
        }

    biases = {
        'bc1' : tf.Variable(tf.random_normal([depth_out1])),#bias applied to all feature maps 
        'bc2' : tf.Variable(tf.random_normal([depth_out2])),#bias applied to all feature maps 
        'bd1' : tf.Variable(tf.random_normal([FCL_Neurons])),#bias applied to all feature maps 
        'out' : tf.Variable(tf.random_normal([n_classes]))#bias applied to all feature maps 
        }
    #TensorFlow Operations
    pred = conv_net(x,weights,biases,keep_prob)
    #Cost Function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels =y))
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #Evaluate
    correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
    accuracy  = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    cost_value = []
    accuracy_value = []
    #Initializing variables
    init = tf.global_variables_initializer()
    #Launching the Execution Graph
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(init)#perform initialization
        for i in range(epochs):
            for j in range(num_batches): #additional loop to break in to mini batches of 256
                batch_x,batch_y = mnist.train.next_batch(batch_size) #get 256 patterns and labels
                sess.run(optimizer,feed_dict={x:batch_x, y:batch_y, keep_prob:dropout}) #randomly drop 25% of neurons in dense layer
                loss,acc = sess.run([cost,accuracy],feed_dict = {x:batch_x,y:batch_y,keep_prob:1.0}) # do not drop neurons during testing keep_prob = 1.0
                cost_value.append(loss)
                accuracy_value.append(acc)
                if epochs % display_step ==0:
                    print("Epoch: ",i+1,"cost: ","{:.9f}".format(loss),"Training Acc: ","{:.5f}".format(acc))
        print("Optimization Completed")
        end_time = time.time()
        print("Total Training Time: ",(end_time-start_time)," secs")
