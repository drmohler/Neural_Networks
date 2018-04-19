import os, shutil
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    keras.__version__

    #Data Preprocessing 
    """
    Currently, our data sits on a drive as JPEG files, so the steps for getting it the images into the network:
    1.Read the picture files.
    2.Decode the JPEG content to RBG grids of pixels.
    3.Convert these into floating point tensors.
    4.Rescale the pixel values (between 0 and 255) to the [0, 1] interval .
    """
    #Setting up the directory paths
    BaseDir = 'P:/2018Spring/ClassCodes/CNN/DogsCats/DogsCats' #Change this to match your folders.
    train_dir = os.path.join(BaseDir,'Train')
    val_dir = os.path.join(BaseDir,'Val')
    test_dir = os.path.join(BaseDir,'Test')
    
    train_dogs_dir = os.path.join(train_dir,'TrainDogs')
    train_cats_dir = os.path.join(train_dir,'TrainCats')
    
    val_dogs_dir = os.path.join(val_dir,'ValDogs')
    val_cats_dir = os.path.join(val_dir,'ValCats')

    test_dogs_dir = os.path.join(test_dir,'TestDogs')
    test_cats_dir = os.path.join(test_dir,'TestCats')

    #Check directory paths
    print("Total Training Dog Images = ",len(os.listdir(train_dogs_dir)))
    print("Total Validation Dog Images = ",len(os.listdir(val_dogs_dir)))
    print("Total Test Dog Images = ",len(os.listdir(test_dogs_dir)))

    print("Total Training Cat Images = ",len(os.listdir(train_cats_dir)))
    print("Total Validation Cat Images = ",len(os.listdir(val_cats_dir)))
    print("Total Test Cat Images = ",len(os.listdir(test_cats_dir)))

    #Using ImageDataGenerator setup convert jpeg image to data with scaling.
    

    #Examining the output of training generator
    

    #Model Architecture
   

    #Fully Connected or Densely Connected Classifier Network
    

    #Output layer with a single neuron since it is a binary class problem
   

    #Configure the model for running
    

    #Train the Model: Fit the model to the Train Data using a batch generator
    
   

    #Saving the Trained Model
   

    #Plotting the loss and accuracy
    """acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs,acc,'b', label = 'Training Accuracy')
    plt.plot(epochs,val_acc,'r',label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    #plt.imshow()
    plt.show()

    plt.plot(epochs,loss,'b', label = 'Training loss')
    plt.plot(epochs,val_loss,'r',label = 'Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    #plt.imshow()
    plt.show()"""

    
    





