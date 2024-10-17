# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:33:59 2021

Exemplo de Rede LSTM extraído do capítulo 8 do livro:
Long Short-Term Memory Networks with Python - Jason Brownlee

A CNN LSTM can be defined by adding CNN layers on the front end 
followed by LSTM layers with a Dense layer on the output.
It is helpful to think of this architecture as defining two 
sub-models: the CNN Model for feature extraction and the 
LSTM Model for interpreting the features across time steps.
"""

print(__doc__)

# Importing libraries and solving dependencies
#import pandas as pd
import numpy as np
from math import sqrt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import LSTM, TimeDistributed, Dense, Reshape
#from keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(2)

# Habilitando a utilização de avisos como erros
np.seterr(all = "raise")

# plt.rcParams["figure.figsize"] = [6.00, 6.00]
# plt.rcParams["figure.autolayout"] = True


# Preparing data to be plotted
def summing_data_3D (data):
    toSum = np.empty([len(data[0]),len(data[0])])
    for i in range(len(data[0])):
        for j in range(len(data[0])):
            toSum[i,j] = np.sum(data[:,i,j])
    return toSum

# Functions to vectorize data RMS calculation
def RMSfunc (data, it, n_features):
    try:
        RMS = np.sqrt((np.sum(np.square(data[it*n_features:it*n_features+n_features]))/n_features))
    except:
        print('\niterator =', it)
    return RMS 

def RMSfuncSW (data, it, n_features):
    return np.sqrt((np.sum(np.square(data[it:n_features+it]))/n_features))

# Functions to vectorize data KURTOSIS calculation
def KUMeanStdfunc (data, it, n_features):
    mean4Data = np.mean(data[it:n_features+it])
    std4Data = np.std(data[it:n_features+it])
    den = (n_features-1)*np.power(std4Data, 4)
    return mean4Data, std4Data, den

def KUxi_u4func (data, mean4Data, jt, n_features):
    xi_u4 = np.power(data[jt]-mean4Data[int(jt/n_features)], 4)
    return xi_u4

def KUfunc (xi_u4, den, kt, n_features):
    KU = np.sum(xi_u4[kt:n_features+kt])/den[int(kt/n_features)]
    return KU
#--------------------------------------------------------------------

# fix random seed for reproducibility
np.random.seed(2)


#***
#***
#***
path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/Pré processamento/DATA/FEMTO_Git/WAVELET_absflat_Ber1_1.npy'

# Load the dataset and prepare it
#dataframe = pd.read_csv("DATA/STFT_flatten.csv", usecols=[1], engine='python')
#dataset = dataframe.values.astype('float32')
dataset = np.load(path).reshape(-1,1)

#***
#***
#***
# Seting the hyperparameters of the algorithm
window = 64                     # Linear dimension of the frame
frame_size = int(sqrt(window))  # first dimension of the frame
n_features = 64                 # size of the preprocessing window

netEpochs = 5                   # Number of epochs to train Net
netBatch = 1024                 # Number of batchs to train Net
ConvFilters = (2, 2)            # Filter for Convolution Net
ConvPool = (2, 2)               # Pool size for Convolutional Net
LSTMCells = frame_size*16       # number of LSTM Cells in Net

#------------------------------------------------------------
#Preprocessing data to feed CNN-LSTM network (RMS e Kurtosi)
#------------------------------------------------------------
choice = int(input('\nChoose a technique to preprocess the data - (1) RMS, (2) RMS*, (3) Kurtosis or (4) None: '))
data = np.reshape(dataset, (-1,))
if choice == 1:
    print('\nCalculating RMS from each data window...')
    it = np.arange(((len(data)/n_features)), dtype=int)
    myRMS = np.vectorize(RMSfunc, signature='(x),(),()->()')
    newDataset = myRMS(data, it, n_features)
elif choice == 2:
        print('\nCalculating RMS* from a sliding window (1 step at a time)')
        it = np.arange((len(data)-n_features), dtype=int)
        myRMS = np.vectorize(RMSfuncSW, signature='(x),(),()->()')
        newDataset = myRMS(data, it, n_features)
elif choice == 3:
        print('\nCalculating Kurtosis from each data window...')
        it = np.arange(0, len(data), n_features, dtype=int)
        jt = np.arange(len(data), dtype=int)
        kt = np.arange(0, len(data), n_features, dtype=int)

        myKUMeanStd = np.vectorize(KUMeanStdfunc, signature='(x),(),()->(),(),()')
        mean4Data, std4Data, den = myKUMeanStd(data, it, n_features)

        myKUxi_u4 = np.vectorize(KUxi_u4func, signature='(x),(y),(),()->()')
        xi_u4 = myKUxi_u4(data, mean4Data, jt, n_features)

        myKurtosis = np.vectorize(KUfunc, signature='(x),(y),(),()->()')
        KU = myKurtosis(xi_u4, den, kt, n_features)
        newDataset = np.copy(KU)
        
elif choice == 4:
    print('\nNo preprocessing technique was used, data remains the same')
    newDataset = np.copy(dataset)

else:
    print('\nYou chose a wrong number')
    print('\nNo preprocessing technique can be used, data remains the same')
    newDataset = np.copy(dataset)

#--------------------------Adjusting DATA---------------------------
print('\nWait... testing data to see if matches frame size...')
testData = len(newDataset)%(frame_size**2)
if testData != 0:
    print('\nData is not compatible with frame size... Wait until adjusting it...')
    bestLenData = int(len(newDataset)/(frame_size**2))
    newLenData = bestLenData*(frame_size**2)
    beginData = len(newDataset)-newLenData
    data4Reshape = newDataset[beginData:]
    print('\nData is read to be reshaped!')
else:
    print('\nData length match frame size!')
    data4Reshape = np.copy(newDataset)


#-----------------------Multidimensional PLOT------------------------
# Preparing data...
dataReshapePlot = np.reshape(data4Reshape, (-1,frame_size,frame_size))
dataPlot = summing_data_3D(dataReshapePlot)

# Create figure and add axis
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111, projection='3d')
# Create meshgrid
X, Y = np.meshgrid(np.linspace(1, frame_size, num=frame_size), 
                   np.linspace(1, frame_size, num=frame_size))
# Plot surface
plot = ax.plot_surface(X=X, Y=Y, Z=dataPlot, cmap='viridis')
# Adjust plot view
ax.view_init(elev=30, azim=225)
ax.dist=11
# Add colorbar
cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
# Set tick marks
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
# Set axis labels
ax.set_title('3D Heatmap of Wavelet Frames of Scale lanes')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Wavelet Scales lanes (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(770, 830)
# plt.savefig('Figures/WAVELET_4D_Learn.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlot, shading='auto')
plt.title('2D Heatmap - Wavelet Magnitude Frames')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/WAVELET_2D_Learn.png') # saving the plot
plt.show()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Scaling the pre-processed dataset
data4scaler = np.reshape(data4Reshape, (-1,1))
scaler = MinMaxScaler(feature_range=(0, 1))
datasetScaled = scaler.fit_transform(data4scaler)

# Split into train and test sets frames
dataReshapeFrame = np.reshape(datasetScaled, (-1,frame_size,frame_size))
train_size = int(len(dataReshapeFrame) * 0.7)
test_size = len(dataReshapeFrame) - train_size
train, test = dataReshapeFrame[0:train_size,:,:], dataReshapeFrame[train_size:len(dataReshapeFrame),:,:]


# Reorganizing the data into X=frame(t) and Y=frame(t+n)
trainXStep, trainYStep = train[0:(len(train)-1),:,:], train[1:,:,:]
testXStep, testYStep = test[0:(len(test)-1),:,:], test[1:,:,:]


# Configuring the parameters of the input data for CNN-LSTM Network 
time_steps = 1
channels = 1

#--------------------------------------------------------------------

# Reshape data as [samples, timesteps, width, height, channels]
trainX = np.reshape(trainXStep, (-1,time_steps, frame_size,frame_size,channels))
trainY = np.reshape(trainYStep, (-1,frame_size,frame_size,channels))
testX = np.reshape(testXStep, (-1,time_steps, frame_size,frame_size,channels))
testY = np.reshape(testYStep, (-1,frame_size,frame_size,channels))

#--------------------------CNN-LSTM Network--------------------------
# define the model
model = Sequential()
model.add(TimeDistributed(Conv2D(frame_size, ConvFilters, activation="relu"),input_shape=(None,frame_size,frame_size,channels)))
model.add(TimeDistributed(MaxPooling2D(pool_size=ConvPool)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(LSTMCells))
model.add(Dense((window), activation="relu"))
model.add(Reshape((frame_size,frame_size,channels)))
model.compile(loss='mean_squared_error', optimizer="adam")
# my_callbacks = [
#       callbacks.EarlyStopping(monitor = 'loss',
#                               min_delta = 0,
#                               patience = 3,
#                               verbose = 0,
#                               restore_best_weights = True),
#       callbacks.ModelCheckpoint(filepath = 'Models/CNN-LSTM_model.{epoch:02d}-{loss:.2f}.h5',
#                                 monitor = 'loss',
#                                 verbose = 0,
#                                 save_best_only = True,
#                                 mode = 'auto',
#                                 save_freq = 1),
#       callbacks.TensorBoard(log_dir='./Logs'),
# ]
print('\n', model.summary(), end='\n')
print('\nTraining the Net...', end='\n')
model.fit(trainX, trainY, epochs=netEpochs, batch_size=netBatch, verbose=1)#, callbacks=my_callbacks)

#--------------------------------------------------------------------

# evaluate model
print('\nEvaluating the Net...', end='\n')
loss = model.evaluate(testX, testY, verbose=1)
print("\nTrain loss: %f" % loss, end='\n')

# make predictions
print('\nMaking predictions......', end='\n')
trainPredict = model.predict(trainX, verbose=1)
testPredict = model.predict(testX, verbose=1)


#**************************************************************
# reshaping predict sets to return to the original scaled values
trainPredictReshaped = np.reshape(trainPredict, (-1,1))
testPredictReshaped = np.reshape(testPredict, (-1,1))

# invert predictions
trainPredictPlot = scaler.inverse_transform(trainPredictReshaped)
testPredictPlot = scaler.inverse_transform(testPredictReshaped)
#***************************************************************

# Forming new frames
dataPlotPTrain = np.reshape(trainPredictPlot, (-1,frame_size,frame_size))
dataPlotPTest = np.reshape(testPredictPlot, (-1,frame_size,frame_size))
dataPlotPTotal = np.vstack((dataPlotPTrain, dataPlotPTest))
dataPlotPTrainSum = summing_data_3D(dataPlotPTrain)
dataPlotPTestSum = summing_data_3D(dataPlotPTest)
dataPlotPTotalSum = summing_data_3D(dataPlotPTotal)


# Ploting predictions (trainPredict)
# Create figure and add axis
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111, projection='3d')
# Create meshgrid
X, Y = np.meshgrid(np.linspace(1, frame_size, num=frame_size), 
                   np.linspace(1, frame_size, num=frame_size))
# Plot surface
plot = ax.plot_surface(X=X, Y=Y, Z=dataPlotPTrainSum, cmap='plasma')
# Adjust plot view
ax.view_init(elev=30, azim=225)
ax.dist=11
# Add colorbar
cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
# Set tick marks
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
# Set axis labels
ax.set_title('3D Heatmap of TRAINED Wavelet Amplitude Frames of Scales lanes after prediction')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Sacles lanes Amplitude (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(0, 750)
# plt.savefig('Figures/WAVELET_Train_4D_Learn.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlotPTrainSum, shading='auto', cmap='plasma') #, vmin=0, vmax=0.5)
plt.title('2D Heatmap - "Train Predict"')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/WAVELET_Heatmap_Train.png') # saving the plot
plt.show()


# Ploting predictions (testPredict) 
# Create figure and add axis
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111, projection='3d')
# Create meshgrid
X, Y = np.meshgrid(np.linspace(1, frame_size, num=frame_size), 
                   np.linspace(1, frame_size, num=frame_size))
# Plot surface
plot = ax.plot_surface(X=X, Y=Y, Z=dataPlotPTestSum, cmap='plasma')
# Adjust plot view
ax.view_init(elev=30, azim=225)
ax.dist=11
# Add colorbar
cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
# Set tick marks
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
# Set axis labels
ax.set_title('3D Heatmap of TESTED Wavelet Amplitude Frames of Scales lanes after prediction')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Scales lanes Amplitude (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(0, 750)
# plt.savefig('Figures/WAVELET_Test_4D_Learn.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlotPTestSum, shading='auto', cmap='plasma') #, vmin=0, vmax=0.5)
plt.title('2D Heatmap - "Test Predict"')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/WAVELET_Heatmap_Test.png') # saving the plot
plt.show()

# Ploting predictions (TOTALPredict)
# Create figure and add axis
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111, projection='3d')
# Create meshgrid
X, Y = np.meshgrid(np.linspace(1, frame_size, num=frame_size), 
                   np.linspace(1, frame_size, num=frame_size))
# Plot surface
plot = ax.plot_surface(X=X, Y=Y, Z=dataPlotPTotalSum, cmap='viridis')
# Adjust plot view
ax.view_init(elev=30, azim=225)
ax.dist=11
# Add colorbar
cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
# Set tick marks
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
# Set axis labels
ax.set_title('3D Heatmap of Wavelet Ampl. Frames of Scales lanes after prediction')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Scales lanes Amplitude (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(300, 1100)
# plt.savefig('Figures/WAVELET_Total.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlotPTotalSum, shading='auto') #, vmin=0, vmax=0.5)
plt.title('2D Heatmap - "TOTAL"')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/WAVELET_Heatmap_Total.png') # saving the plot
plt.show()

#--------------------------------------------------------------------
# prediction on new data
print('\nPredicting on new data...', end='\n')
yhat = model.predict(trainX[-1:,:], verbose=1)
expected = trainX[-1:,:].reshape(-1,1)
expected = expected[-256:,:]
predicted = yhat.reshape(-1,1)
trainScore = sqrt(mean_squared_error(expected, predicted))
print('\nPredict Score: %.2f RMSE' % (trainScore), end='\n')


#--------------------------------------------------------------------
# Plotting the original and autorregression data
flattenDataTotal = np.reshape(dataPlotPTotal, (-1,1))

# Adjusting data to improve plot performance 
sizePlot = 0.1            # size of the last interval to be plotted
dataset4Plot = dataset.astype('float16')
xDataSize = int((len(dataset4Plot)*sizePlot))
xData = np.arange(0, xDataSize)
flatDataTotal4Plot = flattenDataTotal.astype('float16')
xFlatSize = int((len(flatDataTotal4Plot)*sizePlot))
xFlat = np.arange(0, xFlatSize)

fig, axs = plt.subplots(2, 1, figsize=[18.00, 6.00])
fig.suptitle('Scaleogram of Vibration Signal and Autoregression from CNN-LSTM (last %.d samples of each)' % xDataSize, fontsize=20)
axs[0].plot(xData, dataset4Plot[-xDataSize:], label="Rick Wavelet Transformed Vibration Signal X")
axs[0].set_ylabel('Amplitude', fontsize=15)
axs[0].legend(loc="upper left", fontsize=12)
axs[1].plot(xFlat, flatDataTotal4Plot[-xFlatSize:], color='orange', label='CNN-LSTM induced model of RMS data from transformed dataset')
axs[1].set_ylabel('Amplitude', fontsize=15)
axs[1].set_xlabel('Time steps', fontsize=15)
axs[1].legend(loc="upper left", fontsize=12)
# plt.savefig('Figures/Autoregression_Wavelet.png') # saving the plot
plt.show()

#--------------------------------------------------------------------
# Finding the maximum energy of periodogram
maxCoeffLearn = np.amax(flattenDataTotal)
resultMaxCoeffLearn = np.where(flattenDataTotal == np.amax(flattenDataTotal))
print('\nThis is the maximum Scale Amplitude of scaleogram: %.6f' % maxCoeffLearn)
print('\nThis is the sample point of maximum Scale Amplitude of scaleogram: %.0f' % resultMaxCoeffLearn[0])

#***
#***
#***
#--------------------------------------------------------------------
# Saving data into a numpy binary file
np.save('DATA/CNN-LSTM_Ber1_1_Wave.npy', flattenDataTotal)
