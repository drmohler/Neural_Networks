import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
from matplotlib import pyplot as plt

#Reading Input Data
def LoadData(DataFile):
    DataFile = 'jena_climate_2009_2016.csv'
    file = open(DataFile)
    data = file.read();
    file.close()

    lines = data.split('\n') #split csv file by rows
    header = lines[0].split(',') #get the column labels
    lines = lines[1:] #take all data except the time stamp (i.e. the 0th col)
    print(header)
    print("The number of Input Patterns",len(lines))
    
    #Converting the input data into numpy arrays

    #axis 0 goes through time steps
    #axis 1 goes across columns 

    temp = np.zeros((len(lines),len(header) - 1))
    for i,line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        temp[i,:] = values
    return temp #temp is a matrix of all of the input data

def PlotTemperature(temp):
    t = temp[:,1]
    plt.plot(range(len(t)),t) #Plot of the temperature for 1 year
    plt.show()
    plt.plot(range(1440),t[:1440]) #Plot of temperature for ten days (Temperature reading every ten minutes)
    plt.show()

def PlotLoss(train_loss,val_loss):
    epochs = range(len(train_loss))
    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    #this is an iterative function, can be modified to do train/test split 
def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets #yield is what allows the iterative quality LOOK AT THIS

if __name__ == '__main__':
    data = LoadData('jena_climate_2009_2016.csv')
    PlotTemperature(data)

    #Preparing the Data
    """Parameter Values
    lookback = 1440 Observations used: 10 days back
    steps: 6 Data sampled at one data point per hour
    delay: 144 Targets will be 24 hours in future
    """
    look_back = 1440
    steps = 6
    delay = 144 # equivalent to a single days worth of samples 
    batchSize = 128

    #Preprocessing Data (z-scale data for first half and leave other half for testing) 
    mean = data[:200000].mean(axis=0)   #The first 200,000 samples are used for training out of the 420,551 samples
    data -= mean
    std = data[:200000].std(axis = 0)
    data /= std

    #use the mean and std deviation from the training data to then normalize the rest of the data (as shown above) 

    #Training Data
    train_gen = generator(data,lookback= look_back, delay = delay, min_index=0,
                          max_index=200000,shuffle = True,step=steps,batch_size=batchSize)



    #Validation Data
    val_gen = generator(data,lookback= look_back, delay = delay, min_index=200001,
                          max_index=300000,shuffle = True,step=steps,batch_size=batchSize)

    
    #Test Data
    test_gen = generator(data,lookback= look_back, delay = delay, min_index=300001,
                          max_index=None,shuffle = True,step=steps,batch_size=batchSize)

    #Steps
    val_steps = (300000-200001-look_back)//batchSize 
    test_steps = (len(data)-300001-look_back)//batchSize


    #Basic Dense Network
    #Model Architecture

    """model = Sequential()
    model.add(layers.Flatten(input_shape=(look_back//steps,data.shape[-1])))
    model.add(layers.Dense(32,activation='relu'))# single hidden layer with 32 neurons
    model.add(layers.Dense(1))# want actual temp so don't use activation 
    model.compile(optimizer=RMSprop(),loss='mae')#use absolute error since temp can be both + and -
    history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,
                        validation_data=val_gen,validation_steps=val_steps)

    

    #Plotting the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)
    """

    #Recurrent with GRU Cells
    #Model Architecture
    '''model = Sequential()
    model.add(layers.GRU(32,input_shape=(None,data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer = RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,steps_per_epoch=500,epochs = 20,
                                 validation_data= val_gen, validation_steps=val_steps)

    #Plotting the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)'''

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)
    """
    #Recurrent with GRU Cells and dropouts
    #Model Architecture
    '''model = Sequential()
    model.add(layers.GRU(32,dropout=0.2,recurrent_dropout = 0.2 ,input_shape=(None,data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer = RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,steps_per_epoch=500,epochs = 20,
                                 validation_data= val_gen, validation_steps=val_steps)

    #Plotting the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)'''
    
    #Using LSTM
    #Model Architecture
    model = Sequential()
    model.add(layers.LSTM(32,input_shape=(None,data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer = RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,steps_per_epoch=500,epochs = 20,
                                 validation_data= val_gen, validation_steps=val_steps)

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)
    """