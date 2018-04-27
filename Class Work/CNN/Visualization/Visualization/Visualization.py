#This program will demonstrate how to visualize the outputs of theconvolution and max pool layers
import os, shutil
import keras
from keras import layers
from keras import models
from keras.models import load_model #Library for loading saved models.
from keras.preprocessing import image #Library for preprocessing the image into a 4D tensor
import numpy as np
import matplotlib.pyplot as plt

#Loading the saved model
model = load_model('DogsCats.h5')
model.summary()



#Setting up the directory paths to load a test image
BaseDir = 'C:/DRM_NeuralNets/Neural_Networks/Class Work/CNN/DogsCats/DogsCats' #Change this to match your folders.
test_dir = os.path.join(BaseDir,'Test')
test_cats_dir = os.path.join(test_dir,'TestCats')
test_cats_dir = os.path.join(test_cats_dir,'cat.1503.jpg')

test_dogs_dir = os.path.join(test_dir,'TestDogs')
test_dogs_dir = os.path.join(test_dogs_dir,'dog.1503.jpg')

#Loading the test image
img = image.load_img(test_cats_dir,target_size=(150,150))
#img = image.load_img(test_dogs_dir,target_size=(150,150))
img_tensor = image.img_to_array(img)
print(img_tensor.shape)
img_tensor = np.expand_dims(img_tensor,axis=0) # add dimension to beginning
img_tensor /=255.

print(img_tensor.shape)

#Plot the test image
plt.imshow(img_tensor[0])
plt.show()

#Creating a model with one input and eight outputs corresponding to the eight layers in the saved model
layer_outputs = [layer.output for layer in model.layers[:8]]#get the first 8 layers (top 8 layers) only works on conv layers
activation_model = models.Model(input=model.input,output = layer_outputs)

#Provide the input as the test image of the cat and obtain the activations
activations = activation_model.predict(img_tensor) #predict only takes numpy array

#Obtain individual layer activations and display their shape and as images
first_layer_activation = activations[0]
print(first_layer_activation.shape)
'''channel_name = []
for i in range(first_layer_activation.shape[3]):
    plt.matshow(first_layer_activation[0,:,:,i],cmap = 'viridis')#iterate through the feature maps
    channel_name.append(i)
    plt.title(channel_name[i])
    plt.show()'''





#Visualizing every channel in  intermediate layers
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1] #Number of features in the feature map
    size = layer_activation.shape[1]    #Feature map shape = (l,size,size,n_feature) obtain the width

    n_cols = n_features // images_per_row #Tiles of activation channels in the grid
    display_grid = np.zeros((size*n_cols,images_per_row*size)) 

    for col in range(n_cols): #Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:,col*images_per_row+row]
            channel_image -= channel_image.mean() #Post processing to make it visually appealing
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,row*size: (row+1)*size] = channel_image

    scale = 1./size
    plt.figure(figsize = (scale*display_grid.shape[1],scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap = 'viridis')
    plt.show()


