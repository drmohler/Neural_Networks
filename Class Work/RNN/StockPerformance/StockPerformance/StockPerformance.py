import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pylab



def LoadDataSet(FileName):
    dataframe = read_csv(FileName,usecols = [1])
    #plt.plot(dataframe)
    #plt.show();
    return dataframe

def GenerateTrainTestData(data):
    series = data.values.astype('float32')
    scaler = StandardScaler() #helps with Z-Scaling the data
    series = scaler.fit_transform(series)

    #Split the dataset into train and test
    train_size = int(len(series)*0.75)
    test_size = len(series) - train_size
    train,test = series[0:train_size,:],series[train_size:len(series),:]
    return scaler,train,test


def CreateDataSets(series,ts_lag=1):
    dataX = []
    dataY = []
    n_rows = len(series) - ts_lag
    for i in range(n_rows-1):
        a = series[i:(i+ts_lag),0]
        dataX.append(a)
        dataY.append(series[i+ts_lag,0])
    X,Y = np.array(dataX),np.array(dataY)
    return X,Y


if __name__ == '__main__':
    data = LoadDataSet('sp500.csv')
    scaler,Train,Test = GenerateTrainTestData(data)
    trainX,trainY = CreateDataSets(Train,1)
    testX,testY = CreateDataSets(Test,1)

    trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
    testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

    model = Sequential()
    model.add(LSTM(10,input_shape = (1,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_logarithmic_error',optimizer = 'adagrad')

    model.fit(trainX,trainY,epochs=100,batch_size = 30)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
    print('Train Score" %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
    print('Test Score" %.2f RMSE' % (testScore))

    pylab.plot(trainPredict)
    pylab.plot(testPredict)
    pylab.show()



