"""
David R Mohler 
EE 5410: Final Exam (Q1)
Spring 2018

Recognition of Cats and Dogs from spectograms
of audio recordings
"""

import os, shutil
import keras
from keras.layers import *  
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#Setting up the directory paths
BaseDir = os.getcwd()+'/CatsDogsPSD' #current working directory
    
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
input()
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(dim,dim),
                                                    batch_size=(TrainDog+TrainCat),class_mode = 'binary')
val_generator = train_datagen.flow_from_directory(val_dir,target_size=(dim,dim),
                                                    batch_size=(ValDog+ValCat),class_mode = 'binary')
test_generator = train_datagen.flow_from_directory(test_dir,target_size=(dim,dim),
                                                    batch_size=(TestDog+TestCat),class_mode = 'binary')

ImgDim = 200

#Define the model architecture 
model = models.Sequential()

model.add(Conv2D(512,(3,3),activation = 'relu',input_shape = (ImgDim,ImgDim,3)))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(),metrics=['accuracy'])

history = model.fit_generator(train_generator,steps_per_epoch = 1,epochs = 15,
                        validation_data = val_generator,validation_steps = 1)

model.save('Mohler_P1.h5')

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