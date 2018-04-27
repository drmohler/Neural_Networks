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
    model.summary()

    #Setting up the directory paths to load a test image
    BaseDir = 'C:/shapes' #navigate to shapes directory 
    test_dir = os.path.join(BaseDir,'Test')
    test_star_dir = os.path.join(test_dir,'star')
    test_star_dir = os.path.join(test_star_dir,'2480.png')

    test_circle_dir = os.path.join(test_dir,'circle')
    test_circle_dir = os.path.join(test_circle_dir,'2480.png')

    test_tri_dir = os.path.join(test_dir,'triangle')
    test_tri_dir = os.path.join(test_tri_dir,'2480.png')

    test_sq_dir = os.path.join(test_dir,'square')
    test_sq_dir = os.path.join(test_sq_dir,'2480.png')

    imdir = [test_star_dir,test_circle_dir,test_tri_dir,test_sq_dir]

    #test_dogs_dir = os.path.join(test_dir,'TestDogs')
    #test_dogs_dir = os.path.join(test_dogs_dir,'dog.1503.jpg')

    #Loading the test image
    img1 = image.load_img(imdir[0],target_size=(Img_dim,Img_dim),grayscale=True)
    img2 = image.load_img(imdir[1],target_size=(Img_dim,Img_dim),grayscale=True)
    img3 = image.load_img(imdir[2],target_size=(Img_dim,Img_dim),grayscale=True)
    img4 = image.load_img(imdir[3],target_size=(Img_dim,Img_dim),grayscale=True)

    Images = [img1,img2,img3,img4] 
    
    img_tensors = []
    for i in range(4):
        img_tensor = image.img_to_array(Images[i])
        img_tensor = np.expand_dims(img_tensor,axis=0) # add dimension to beginning
        img_tensor /=255.
        img_tensors.append(img_tensor)
    
    
    #Display test images (1 of each shape) 
    f,axarr = plt.subplots(1,4)
    axarr[0].imshow(Images[0])
    axarr[1].imshow(Images[1])
    axarr[2].imshow(Images[2])
    axarr[3].imshow(Images[3])
    plt.show()
    
    
    layer_outputs = [layer.output for layer in model.layers[:num_layers]]#get the first 2 layers
    activation_model = models.Model(input=model.input,output = layer_outputs)

    #Provide the input as the test image of the cat and obtain the activations
    act_list = []
    for i in range(4):
        activations = activation_model.predict(img_tensors[i]) #predict only takes numpy array
        act_list.append(activations) 

    #Obtain individual layer activations and display their shape and as images
    FLact = []
    for i in range(4): 
        first_layer_activation = act_list[i][0]
        FLact.append(first_layer_activation)
    
    '''channel_name = []
    for i in range(first_layer_activation.shape[3]):
        plt.matshow(first_layer_activation[0,:,:,i],cmap = 'viridis')#iterate through the feature maps
        channel_name.append(i)
        plt.title(channel_name[i])
        plt.show()'''

    #Visualizing every channel in  intermediate layers
    layer_names = []
    for layer in model.layers[:num_layers]:
        layer_names.append(layer.name)

    images_per_row = 2 
    for i in range(4): #loop over all four shapes
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
                    channel_image = np.clip(channel_image,0,255).astype('uint8')
                    display_grid[col*size:(col+1)*size,row*size: (row+1)*size] = channel_image

            scale = 1./size
            plt.figure(figsize = (scale*display_grid.shape[1],scale*display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid,aspect='auto',cmap = 'viridis')
            plt.show()


