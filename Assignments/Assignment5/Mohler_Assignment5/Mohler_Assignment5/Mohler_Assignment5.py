
import os, shutil
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import Visualization as vis #use to visualize network layers


class ShapeNet1: 
    @staticmethod
    def build(dim): #dim is the square dimensions of the input image
        model = models.Sequential()

        model.add(layers.Conv2D(4,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.AvgPool2D((2,2)))

        model.add(layers.Conv2D(4,(3,3),activation = 'relu'))
        model.add(layers.AvgPool2D((2,2)))

        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5)) 
        model.add(layers.Dense(8,activation='relu'))

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary()
        num_layers = 4
        return model,num_layers

class ShapeNet2: #has issues with divergent training (i.e. hit or miss) 
    @staticmethod
    def build(dim):
        model = models.Sequential()

        model.add(layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.MaxPool2D((2,2)))

        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8,activation='relu'))

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary()

        num_layers = 2
        return model,num_layers


class ShapeNet3: 
    @staticmethod
    def build(dim):
        model = models.Sequential()

        model.add(layers.Conv2D(4,(3,3),activation = 'relu',input_shape = (dim,dim,1)))
        model.add(layers.MaxPool2D((2,2)))

        model.add(layers.Conv2D(2,(3,3),activation = 'relu'))
        model.add(layers.MaxPool2D((2,2)))

        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten())
        #model.add(layers.Dropout(0.5)) 
        #model.add(layers.Dense(8,activation='relu'))

        #Output layer with softmax classifier to for multiclass-single label problem
        model.add(layers.Dense(4,activation='softmax'))
        model.summary()
        
        num_layers = 2
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
    BaseDir = "C:/shapes" #Change this to match your folders.
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

    #Check directory paths
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
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True) 
    val_datagen = ImageDataGenerator(rescale= 1./255)
    test_datagen = ImageDataGenerator(rescale= 1./255)

    dim = 50

    #binary mode create a 1D tensor with only 2 values
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

    epochs = 15
    LR = 1e-2

    #recommended epochs: 20 
    # conv(4)->conv(4)->flatten->dense(8)->dense(4)
    model,num_layers = ShapeNet1.build(dim) 

   
    #recommended epochs: 5 
    # conv(16)->flatten->dense(8)->dense(4)
    #model,num_layers = ShapeNet2.build(dim)
    '''NOTE: often diverges to random chance (25%)'''

    #recommended epochs: 25
    # conv(4)->conv(3)->flatten->dense(4)
    #model,num_layers = ShapeNet3.build(dim)
    
   
    #Configure the model for running
    model.compile(loss='categorical_crossentropy',optimizer = optimizers.RMSprop(lr=LR)
                  ,metrics=['accuracy'])

    #Train the Model: Fit the model to the Train Data using a batch generator
    history = model.fit_generator(train_generator,steps_per_epoch = 248,epochs = epochs,
                        validation_data = validation_generator,validation_steps = 248)#(1240*3)/20
    
    Loss,Accuracy  = model.evaluate_generator(test_generator)

    print("Network Accuracy (Test): ",Accuracy)
   
    #Saving the Trained Model

    model_file = 'Shapes1.h5'
    model.save(model_file)

    vis.visualize(model_file,dim,num_layers) #run layer visualizations


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

