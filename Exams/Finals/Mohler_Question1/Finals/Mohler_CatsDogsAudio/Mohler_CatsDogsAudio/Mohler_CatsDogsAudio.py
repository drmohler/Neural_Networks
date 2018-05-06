"""
David R Mohler 
EE 5410: Final Exam (Q1)
Spring 2018

Recognition of Cats and Dogs from spectograms
of audio recordings
"""

import os, shutil
import keras
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout 
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import Visualization as vis #import the visualization file 


class Net1(): #current best is 86.36% accurate 
    @staticmethod
    def build():
        model = models.Sequential()

        model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (dim,dim,3)))
        model.add(MaxPool2D((2,2)))

        model.add(Conv2D(64,(3,3),activation = 'relu'))
        model.add(MaxPool2D((2,2)))

        model.add(Conv2D(32,(3,3),activation = 'relu'))
        model.add(MaxPool2D((2,2)))

        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation = 'sigmoid'))
        model.summary()
        num_layers = 6
        return model, num_layers

class Net2(): #current best is 
    @staticmethod
    def build():
        model = models.Sequential()

        model.add(Conv2D(16,(3,3),activation = 'relu',input_shape = (dim,dim,3)))
        model.add(MaxPool2D((2,2)))

        model.add(Conv2D(24,(3,3),activation = 'relu'))
        model.add(MaxPool2D((2,2)))

        model.add(Conv2D(36,(3,3),activation = 'relu'))
        model.add(MaxPool2D((2,2)))

        model.add(Conv2D(16,(3,3),activation = 'relu'))
        model.add(MaxPool2D((2,2)))

        model.add(Flatten())

        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation = 'sigmoid'))
        model.summary()
        num_layers = 8
        return model, num_layers

#Setting up the directory paths
BaseDir = 'C:/CatsDogsPSD' #use dataset at root of C: drive
    
train_dir = os.path.join(BaseDir,'train')
val_dir = os.path.join(BaseDir,'val')
test_dir = os.path.join(BaseDir,'test')
    
train_dogs_dir = os.path.join(train_dir,'dog')
train_cats_dir = os.path.join(train_dir,'cat')
    
val_dogs_dir = os.path.join(val_dir,'dog')
val_cats_dir = os.path.join(val_dir,'cat')

test_dogs_dir = os.path.join(test_dir,'dog')
test_cats_dir = os.path.join(test_dir,'cat')

#Check directory paths

TrainDog = len(os.listdir(train_dogs_dir)) #store the number of dog train images
ValDog   = len(os.listdir(val_dogs_dir))
TestDog  = len(os.listdir(test_dogs_dir))

TrainCat = len(os.listdir(train_cats_dir)) #store the number of dog train images
ValCat   = len(os.listdir(val_cats_dir))
TestCat = len(os.listdir(test_cats_dir))

print("Total Training Dog Images = ",TrainDog)
print("Total Validation Dog Images = ",ValDog)
print("Total Test Dog Images = ", TestDog)

print("Total Training Cat Images = ",TrainCat)
print("Total Validation Cat Images = ",ValCat)
print("Total Test Cat Images = ",TestCat)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True,
                                   width_shift_range = 0.2)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

dim = 200

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(dim,dim),
                                                    batch_size=10,class_mode = 'binary')
val_generator = train_datagen.flow_from_directory(val_dir,target_size=(dim,dim),
                                                    batch_size=10,class_mode = 'binary')
test_generator = train_datagen.flow_from_directory(test_dir,target_size=(dim,dim),
                                                    batch_size=10,class_mode = 'binary')
'''(TrainDog+TrainCat)'''
'''(ValDog+ValCat)'''
'''(TestDog+TestCat)'''

#Define the model architecture 
model , num_layers = Net2.build()

model.compile(loss='binary_crossentropy',optimizer=optimizers.Adamax(),metrics=['accuracy'])


history = model.fit_generator(train_generator,steps_per_epoch = (TrainDog+TrainCat)//10,epochs = 15,
                        validation_data = val_generator,validation_steps = (ValDog+ValCat))

test_loss, test_acc = model.evaluate_generator(test_generator) #pass the test set through the trained
                                                               #model to get test accuracy 

print("Network Accuracy (TEST): ", test_acc)

filename = 'Mohler_P1.h5'
model.save(filename)

vis.visualize(filename,dim,num_layers)

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