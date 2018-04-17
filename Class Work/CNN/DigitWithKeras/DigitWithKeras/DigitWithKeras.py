import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

if __name__ == '__main__':
    keras.__version__

    (train_images,train_labels), (test_images,test_labels) = mnist.load_data()
    print("The number of images in training set = ",train_images.shape[0])
    print("The number of images in test set = ",test_images.shape[0])
    print("The dimension of each training pattern = ",train_images.shape[1])
    
    #Shaping training and test images
    

    #Model Architecture
    

    #Fully Connected or Densely Connected Classifier Network
   
    #Configure the model for running
   

    #Train the Model: Fit the model to the Train Data
    
    #Save the Model
    
    #Evaluate the Model
   




