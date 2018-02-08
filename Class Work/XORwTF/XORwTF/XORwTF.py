#Application to implement exclusive OR with tensor flow

import tensorflow as tf

#Define Variables in TF 
#Define placeholder for a variable which will be initialized later

#each row in matrix represents 1 input pattern
# --> need to know # of cols but rows is variable

#define float array known as tensor
x = tf.placeholder(tf.float32,shape = [4,2],name="x-input")   #input (type: single precision, shape of input [rows,cols] or [None,cols],name)
y = tf.placeholder(tf.float32,shape = [4,1],name="y-output")   

#setting up initialized weight tensors need 2 for xor, hidden and output
w1 = tf.Variable(tf.random_uniform([2,2],-1,1,name="Weights1")) #(count number of lines to each neuron and # of neurons)-->[2,2]
w2 = tf.Variable(tf.random_uniform([2,1],-1,1,name="Weights2"))

#setting up bias
b1 = tf.Variable(tf.zeros([2],name="bias1"))
b2 = tf.Variable(tf.zeros([1],name="bias2"))

#Defining the final output in the feed forward (FF) direction 
z2 = tf.sigmoid(tf.matmul(x,w1)+b1)
pred =tf.sigmoid(tf.matmul(z2,w2)+b2)
if __name__ == '__main__':

    print("Doing things and stuff")