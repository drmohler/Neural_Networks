import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt

if __name__ == '__main__':
    keras.__version__

    (train_images,train_labels), (test_images,test_labels) = mnist.load_data()
    print("The number of images in training set = ",train_images.shape[0])
    print("The number of images in test set = ",test_images.shape[0])
    print("The dimension of each training pattern = ",train_images.shape[1])
    
    #Shaping training and test images
    
    train_images = train_images.reshape(60000,28,28,1)
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape(10000,28,28,1)
    test_images = test_images.astype('float32')/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #Model Architecture
    model = models.Sequential() # creatse a model cpable of linearly stacked layers

    #convolution layer
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
    model.add(layers.MaxPool2D((2,2)))

    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))

    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    

    #Fully Connected or Densely Connected Classifier Network
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.summary() #shows parameters of each layer   
    #Configure the model for running
    model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    #'accuracy' is a print name for the console, it is actually stored as 'acc' in a dictionary

    #Train the Model: Fit the model to the Train Data
    H = model.fit(train_images,train_labels,epochs = 5,batch_size=64)

    
    #Save the Model
    model.save('MNISTwKeras.h5')
    #Evaluate the Model
    test_loss,test_acc = model.evaluate(test_images,test_labels)
    print("Loss Value for Test Images: ",test_loss)
    print("Accuracy Value for Test Images: ",test_acc)

    #plotting
    acc = H.history['acc']
    loss = H.history['loss']
    epochs = range(len(acc))

    plt.plot(epochs,acc,'b',label = 'Training Accuracy')
    plt.show()

    plt.plot(epochs,loss,'r',label = 'Training Loss')
    plt.show()




