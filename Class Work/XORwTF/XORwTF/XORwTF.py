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
w1 = tf.Variable(tf.random_uniform([2,2],-1,1),name="Weights1") #(count number of lines to each neuron and # of neurons)-->[2,2]
w2 = tf.Variable(tf.random_uniform([2,1],-1,1),name="Weights2")

#setting up bias
b1 = tf.Variable(tf.zeros([2]),name="bias1")
b2 = tf.Variable(tf.zeros([1]),name="bias2")

#Defining the final output in the feed forward (FF) direction 
z2 = tf.sigmoid(tf.matmul(x,w1)+b1) # in graph, these are operations --> matrix mult, addition, activation func
pred =tf.sigmoid(tf.matmul(z2,w2)+b2) #feed output of 1st layer to 2nd and get output

#Training is done using the gradient approach. i.e. minimize the error
#Develop a cost/loss function depending on the nature of the problem. 
#Training goal: minimize the cost function

cost = tf.reduce_mean(((y*tf.log(pred))+((1-y)*tf.log(1.0-pred)))*-1) #We used sum so if divide by n we get minimal cost func (i.e. mean)
                                                                      #This is also an set of operations
learning_rate = 0.01 #Rate must be small (did not explain why) 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#Each training step, do gradient descent to min the cost function. 
                                                                            #Up to here we have costructed computational graph (Always first step in TF) 

#Set up input data
XOR_X = [[0,0],[0,1],[1,0],[1,1]] #Shape is 4x2, must match shape of placeholder
XOR_Y = [[0],[1],[1],[0]]

#initialize training variables
init = tf.global_variables_initializer()#Takes tf.variable and initializes all of them
sess = tf.Session() #create a session

sess.run(init)#Run the session with initial values
print("Initial weights of hidden layer: ")
print(sess.run(w1))#Displays initial random weights
print("\nInitial weights of ouput layer: ")
print(sess.run(w2))#Displays initial random weights
print("\nInitial bias of hidden layer: ")
print(sess.run(b1))#Displays initial random weights
print("\nInitial bias of output layer: ")
print(sess.run(b2))#Displays initial random weights

#Generate a log of the flow
writer = tf.summary.FileWriter("C:/tflogs",sess.graph)

print("\nStart Training")

for i in range(100000): #100000 epochs
    sess.run(train_step,feed_dict={x:XOR_X,y:XOR_Y})#will only run once, so put it in this loop 
                                                    #run session on train step and replace placeholders with XOR_*
                                                    #Does gradient optimization on ALL inputs and minimizes cost
print("\nTraining COMPLETE")
print("\nweights of hidden layer: ")
print(sess.run(w1))#Displays initial random weights
print("\nweights of ouput layer: ")
print(sess.run(w2))#Displays initial random weights
#to manually verify, also need biases printed

#Results
print("\nFinal Prediction: ",sess.run(pred,feed_dict={x:XOR_X,y:XOR_Y}))#will execute z2 without explicit instruction due to graph structure
writer.close

if __name__ == '__main__':

    print("Doing things and stuff")