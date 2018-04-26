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

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print(header)
    print("The number of Input Patterns",len(lines))
    
    #Converting the input data into numpy arrays

    temp = np.zeros((len(lines),len(header) - 1))
    for i,line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        temp[i,:] = values
    return temp

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
        yield samples, targets

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
    delay = 144
    batchSize = 128

    #Preprocessing Data
    mean = data[:200000].mean(axis=0)   #The first 200,000 samples are used for training out of the 420,551 samples
    data -= mean
    std = data[:200000].std(axis = 0)
    data /= std

    #Training Data
    

    #Validation Data
    
    
    #Test Data
    

    #Steps

    #Basic Dense Network
    #Model Architecture
    

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)"""

    #Recurrent with GRU Cells
    #Model Architecture
    

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)
    """
    #Recurrent with GRU Cells and dropouts
    #Model Architecture
    

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)"""
    
    #Using LSTM
    #Model Architecture
    

    #Plotting the training and validation loss
    """loss = history.history['loss']
    val_loss = history.history['val_loss']
    PlotLoss(loss,val_loss)
    """