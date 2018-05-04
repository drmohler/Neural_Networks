
import os, shutil
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import Visualization as vis #use to visualize network layers

#The classes below define the 3 tested architectures
class ShapeNet1: 
    @staticmethod
    def build(dim): #dim is the square dimensions of the input image
        model = models.Sequential()

        #convolution layer with 4 3x3 kernels, passed the input image
        model.add(layers.Conv2D(4,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.AvgPool2D((2,2))) #perform average pooling 2x2 with a stride of 2 (default) 

        #second convolution layer with 4 3x3 kernels 
        model.add(layers.Conv2D(4,(3,3),activation = 'relu'))
        model.add(layers.AvgPool2D((2,2)))

        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten()) #restructure data for FC network 
        model.add(layers.Dropout(0.5)) #apply dropout regularization
        model.add(layers.Dense(8,activation='relu')) #Dense layer with 8 fully connected neurons

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary() #print the network parameters
        num_layers = 4 #parameter used for plotting activation maps in Visualization
        return model,num_layers

class ShapeNet2: #has issues with divergent training (i.e. hit or miss) 
    @staticmethod
    def build(dim):
        model = models.Sequential()

        #single convolution layer with 16 3x3 kernels, passed the input image
        model.add(layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.MaxPool2D((2,2)))

        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5)) #apply dropout regularization with a probability of 50%
        model.add(layers.Dense(8,activation='relu')) #Dense layer with 8 fully connected neurons

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary()

        num_layers = 2 #parameter used for plotting activation maps in Visualization
        return model,num_layers


class ShapeNet3: 
    @staticmethod
    def build(dim):
        model = models.Sequential()

        #convolution layer with 4 3x3 kernels, passed the input image
        model.add(layers.Conv2D(4,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.MaxPool2D((2,2))) #use maxpooling 2x2 with a stride of 2 

        #second convolution layer with only 2 3x3 kernels 
        model.add(layers.Conv2D(2,(3,3),activation = 'relu'))
        model.add(layers.MaxPool2D((2,2))) 

        #Remove the first fully connected layer and pass directly to softmax layer
        model.add(layers.Flatten())

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary()
        
        num_layers = 4 #parameter used for plotting activation maps in Visualization
        return model,num_layers


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
    BaseDir = "C:/shapes" #the shapes folder needs to be in the root of the C drive
    train_dir = os.path.join(BaseDir,'Train')
    val_dir = os.path.join(BaseDir,'Val')
    test_dir = os.path.join(BaseDir,'Test')
    
    train_circle_dir = os.path.join(train_dir,'circle')
    train_square_dir = os.path.join(train_dir,'square')
    train_star_dir = os.path.join(train_dir,'star')
    train_triangle_dir = os.path.join(train_dir,'triangle')
    
    val_circle_dir = os.path.join(val_dir,'circle')
    val_square_dir = os.path.join(val_dir,'square')
    val_star_dir = os.path.join(val_dir,'star')
    val_triangle_dir = os.path.join(val_dir,'triangle')

    test_circle_dir = os.path.join(test_dir,'circle')
    test_square_dir = os.path.join(test_dir,'square')
    test_star_dir = os.path.join(test_dir,'star')
    test_triangle_dir = os.path.join(test_dir,'triangle')

    #Check directory paths for each type of shape 
    print("\nTotal Training Circle Images = ",len(os.listdir(train_circle_dir)))
    print("Total Validation Circle Images = ",len(os.listdir(val_circle_dir)))
    print("Total Test Circle Images = ",len(os.listdir(test_circle_dir)))

    print("\nTotal Training Square Images = ",len(os.listdir(train_square_dir)))
    print("Total Validation Square Images = ",len(os.listdir(val_square_dir)))
    print("Total Test Square Images = ",len(os.listdir(test_square_dir)))

    print("\nTotal Training Star Images = ",len(os.listdir(train_star_dir)))
    print("Total Validation Star Images = ",len(os.listdir(val_star_dir)))
    print("Total Test Star Images = ",len(os.listdir(test_star_dir)))

    print("\nTotal Training Triangle Images = ",len(os.listdir(train_triangle_dir)))
    print("Total Validation Triangle Images = ",len(os.listdir(val_triangle_dir)))
    print("Total Test Triangle Images = ",len(os.listdir(test_triangle_dir)))
   

    #Using ImageDataGenerator setup convert jpeg image to data with scaling.
    #IDG can have 8-26 arguments
    train_datagen = ImageDataGenerator(rescale= 1./255, # convert pixel integer values
                                       #perfrom some data augmentation due to small data sets
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True) 
    val_datagen = ImageDataGenerator(rescale= 1./255) #use data generator to scale pixel values in validation set
    test_datagen = ImageDataGenerator(rescale= 1./255) #use data generator to scale pixel values in test set

    dim = 50 #set the square dimension to scale the images to (i.e 50x50 input images from the original 200x200) 

    #categorical mode creates a 2D one-hot encoded labels for all three data sets
    #since the images are black and white convert them to grayscale to use only a single channel
    train_generator = train_datagen.flow_from_directory(train_dir, target_size = (dim,dim),
                                                        color_mode="grayscale",batch_size=20,class_mode='categorical') #searches all subfolders for images
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size = (dim,dim),
                                                            color_mode="grayscale",batch_size=20,class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size = (dim,dim),
                                                            color_mode="grayscale",batch_size=20,class_mode='categorical')

    #Examining the output of training generator
    for data_batch, labels_batch in train_generator:
        print("Data Batch Shape: ",data_batch.shape)
        print("Labels Batch Shape: ",labels_batch.shape)
        #print(data_batch[0,0])
        print(labels_batch[0])
        break

  
    #Model Architecture(s) 
    """NOTE: Remove comment blocks below to run different models, 
             also comment out unused architectures""" 

    epochs = 25
    LR = 1e-2

    """
    #------------SHAPENET1-------------------#
    #recommended epochs: 25 
    # conv(4)->conv(4)->flatten->dense(8)->dense(4)
    model,num_layers = ShapeNet1.build(dim) 
    """

    #------------SHAPENET2-------------------#
    """
    #recommended epochs: 15 
    # conv(16)->flatten->dense(8)->dense(4)
    model,num_layers = ShapeNet2.build(dim)
    '''NOTE: often diverges to random chance (25%)'''
    """

    #------------SHAPENET3-------------------#
    
    #recommended epochs: 25
    # conv(4)->conv(3)->flatten->dense(4)
    model,num_layers = ShapeNet3.build(dim)
    
    
       
    #Configure the model for running using categorical cross entropy for multiclass-single label problem
    #use RMS prop optimizer from empirical performance testing
    model.compile(loss='categorical_crossentropy',optimizer = optimizers.RMSprop(lr=LR)
                  ,metrics=['accuracy'])

    #Train the Model: Fit the model to the Train Data using a batch generator
    history = model.fit_generator(train_generator,steps_per_epoch = 248,epochs = epochs,
                        validation_data = validation_generator,validation_steps = 248)#(1240*3)/20 = 248 steps
    
    Loss,Accuracy  = model.evaluate_generator(test_generator) # Calculate the performance of the network on the test images

    print("\nNetwork Accuracy (Test): ",Accuracy)
   
    #Saving the Trained Model

    model_file = 'Shapes1.h5'
    model.save(model_file)

    vis.visualize(model_file,dim,num_layers) #run layer visualizations 
    #(this is imported from a file in the same directory: Visualization.py)


    #Plotting the loss and accuracy
    acc = history.history['acc']
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
    plt.show()

