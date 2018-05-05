#This program will demonstrate how to visualize the outputs of theconvolution and max pool layers
import os, shutil
import keras
from keras import layers
from keras import models
from keras.models import load_model #Library for loading saved models.
from keras.preprocessing import image #Library for preprocessing the image into a 4D tensor
import numpy as np
import matplotlib.pyplot as plt


def visualize(model_file,Img_dim,num_layers):
    #Loading the saved model
    model = load_model(model_file)
    

    #Setting up the directory paths to load a test image
    BaseDir = 'C:/CatsDogsPSD' #navigate to PSD image directory
    test_dir = os.path.join(BaseDir,'test')
    test_dog_dir = os.path.join(test_dir,'dog')
    test_dog_dir = os.path.join(test_dog_dir,'dog_barking_103.jpg') #hardcode an image to visualize

    test_cat_dir = os.path.join(test_dir,'cat')
    test_cat_dir = os.path.join(test_cat_dir,'cat_158.jpg')


    #list of image directories for images to visualize 
    imdir = [test_dog_dir, test_cat_dir]

    #Loading the test image
    img1 = image.load_img(imdir[0],target_size=(Img_dim,Img_dim))
    img2 = image.load_img(imdir[1],target_size=(Img_dim,Img_dim))


    Images = [img1,img2] #creat a list of processed images 
    
    img_tensors = []
    for i in range(len(imdir)):
        img_tensor = image.img_to_array(Images[i])
        img_tensor = np.expand_dims(img_tensor,axis=0) # add dimension to beginning
        img_tensor /=255.
        img_tensors.append(img_tensor)
    
    
    #Display test images (1 of each shape) in a row  
    f,axarr = plt.subplots(1,2)
    axarr[0].imshow(Images[0])
    axarr[1].imshow(Images[1])
    plt.show()
    
    
    layer_outputs = [layer.output for layer in model.layers[:num_layers]]#get the appropriate number of layers for the network 
    activation_model = models.Model(input=model.input,output = layer_outputs)

    #Provide the input as the test image of the shape and obtain the activations
    act_list = []
    for i in range(len(imdir)):
        activations = activation_model.predict(img_tensors[i]) #use network to predict what the patterns should be
        act_list.append(activations) 

    #Obtain individual layer activations and display their shape as images
    FLact = []
    for i in range(len(imdir)): 
        first_layer_activation = act_list[i][0]
        FLact.append(first_layer_activation)
    
    #Visualizing every channel in  intermediate layers
    layer_names = []
    for layer in model.layers[:num_layers]:
        layer_names.append(layer.name)

    images_per_row = 8 #format activation maps in to rows of 2 by num_layers
    for i in range(len(imdir)): #loop over all four shapes
        for layer_name, layer_activation in zip(layer_names,act_list[i]):
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
                    channel_image = np.clip(channel_image,0,255).astype('uint8') #ensure data is in std image range (0,255)
                    display_grid[col*size:(col+1)*size,row*size: (row+1)*size] = channel_image

            scale = 1./size
            plt.figure(figsize = (scale*display_grid.shape[1],scale*display_grid.shape[0]))
            if i == 0:
                plt.title('Dog_'+layer_name)
            else:
                plt.title('Cat_'+layer_name) 
            plt.grid(False)
            plt.imshow(display_grid,aspect='auto',cmap = 'viridis')
            plt.show()


