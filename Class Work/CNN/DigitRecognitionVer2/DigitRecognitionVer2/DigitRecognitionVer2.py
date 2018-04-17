import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

#Function to read the MNIST dataset along with the class labels
def readMNISTData():
    """read_data_sets returns a nested structure of python type objects to each 
    component of an element of the dataset"""
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)  
    return mnist

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))

def conv_net(X,w_conv1,w_conv2,w_conv3,w_fcl,w_output,keep_conv,keep_hidden):
    conv1 = tf.nn.conv2d(X,w_conv1,strides = [1,1,1,1],padding = 'SAME')
    conv1_d = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_d,ksize = [1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
    conv1 = tf.nn.dropout(conv1,keep_conv)

    conv2 = tf.nn.conv2d(conv1,w_conv2,strides = [1,1,1,1],padding = 'SAME')
    conv2_d = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_d,ksize = [1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
    conv2 = tf.nn.dropout(conv2,keep_conv)

    conv3 = tf.nn.conv2d(conv2,w_conv3,strides = [1,1,1,1],padding = 'SAME')
    conv3 = tf.nn.relu(conv3)

    FC_layer = tf.nn.max_pool(conv3,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')
    FC_layer = tf.reshape(FC_layer,[-1,w_fcl.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer,keep_conv)

    output_layer = tf.nn.relu(tf.matmul(FC_layer,w_fcl))
    output_layer = tf.nn.dropout(output_layer,keep_hidden)

    result = tf.matmul(output_layer,w_output)

    return result

if __name__ == '__main__':
    mnist = readMNISTData()
    trainX,trainY,testX,testY = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

    print("The dimension of each training pattern: ",trainX.shape[1])
    print("The number of training patterns: ",trainX.shape[0])
    print("The dimension of each test pattern: ",testX.shape[1])
    print("The number of test patterns: ",testX.shape[0])

    print("The class labels of Training Patterns: ",trainY[0:10])
    print("The number of class labels of Training Patterns: ",trainY.shape)

    print("The number of class labels of Test Patterns: ",testY.shape)

    learning_rate = 0.01
    epochs = 20
    batch_size = 128
    num_batches = int(mnist.train.num_examples/batch_size)
    print("Number of Batches = ",num_batches)
    test_size = 256
    img_size = 28
    num_classes = 10
    display_step = 1

    #Reshaping the input training and test images
    trainX = trainX.reshape(-1,img_size,img_size,1)     #28x28x1 inpur image
    testX = testX.reshape(-1,img_size,img_size,1)       #28x28x1 inpur image

    #Input Output Definition
    x = tf.placeholder(tf.float32,[None,img_size,img_size,1])
    y = tf.placeholder(tf.float32,[None,num_classes])

    #Kernels and Weight Matrices Definition
    w = init_weights([3,3,1,32])    #Kernel of the first convolution layer
    w2 = init_weights([3,3,32,64])  #Kernel of the second convolution layer
    w3 = init_weights([3,3,64,128]) #Kernel of the third convolution layer
    w4 = init_weights([128*4*4,625])    #Weight  Matrix for the Fully Connected Layer
    w_o = init_weights([625,num_classes])   #Weight matrix for the output layer

    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)
    pred = conv_net(x,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)

    #Cost Function
    cost_ = tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels=y)
    cost  = tf.reduce_mean(cost_)

    #Optimizer
    #optimizer = tf.train.RMSPropOptimizer(learning_rate,0.9).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Evaluate
    predict_op = tf.arg_max(pred,1)
    correct_pred = tf.equal(predict_op,tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    cost_value = []
    accuracy_value = []

    #pred_op = tf.arg_max(pred,1)
    #Initializing variables
    init = tf.global_variables_initializer()

    #Launching the Execution Graph
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            training_batch = zip(range(0,len(trainX),batch_size),range(batch_size,len(trainX)+1,batch_size))
            for start,end in training_batch:
                sess.run(optimizer,feed_dict={x:trainX[start:end],y:trainY[start:end],p_keep_conv:0.8,p_keep_hidden:0.5})
                loss,acc = sess.run([cost,accuracy],feed_dict={x:trainX[start:end],y:trainY[start:end],p_keep_conv:1.0,p_keep_hidden:1.0})
                cost_value.append(loss)
                accuracy_value.append(acc)
                if epochs % display_step == 0:
                    print("Epoch: ",(i+1)," cost: ","{:.9f}".format(loss)," Training accuracy: ","{:.5f}".format(acc))

        print("Optimization Completed")
        end_time = time.time()
        print("Total Training Time = ",(end_time-start_time)," secs")

        #Plotting the cost function
        fig2 = plt.figure("Variation of cost function with Number of Epochs")
        plt.plot(cost_value)
        plt.xlabel('Batch')
        plt.ylabel('Cost')
        plt.show()

        #Plotting the Accuracy
        fig3 = plt.figure("Variation of Accuracy with Number of Epochs")
        plt.plot(accuracy_value)
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.show()

        y1 = sess.run(pred,feed_dict={x:testX[:256],p_keep_conv:1.0,p_keep_hidden:1.0}) #no more images, GPU memoryy
        test_classes = np.argmax(y1,1)
        test_accuracy = sess.run(accuracy,feed_dict={x:testX[:256],y:testY[:256],p_keep_conv:1.0,p_keep_hidden:1.0})

        print("Test Accuracy = ",test_accuracy)
        f,a = plt.subplots(1,10,figsize = (10,2))

        print(np.argmax(mnist.test.labels[0:10],axis = 1))
        print(test_classes[0:10])
        for i in range(10):
            a[i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        plt.show()


    

