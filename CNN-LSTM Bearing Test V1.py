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
#from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow import set_random_seed
set_random_seed(2)

# plt.rcParams["figure.figsize"] = [6.00, 6.00]
# plt.rcParams["figure.autolayout"] = True

# Function to Vector a concatenation 
def myfunc(dataset, it, nPredict):
    return dataset[it:nPredict+it]


# Preparing data to be plotted
def summing_data_3D (data):
    toSum = np.empty([len(data[0]),len(data[0])])
    for i in range(len(data[0])):
        for j in range(len(data[0])):
            toSum[i,j] = np.sum(data[:,i,j])
    return toSum

#--------------------------------------------------------------------

# fix random seed for reproducibility
np.random.seed(2)

#***
#***
#***
path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/Pré processamento/DATA/FEMTO_Git/STFT_flat_Ber1_3.npy'

# Load the dataset and prepare it
#dataframe = pd.read_csv("DATA/STFT_flatten.csv", usecols=[1], engine='python')
#dataset = dataframe.values.astype('float32')
dataset = np.load(path).reshape(-1,1)

#***
#***
#***
window = 64                         # Size of STFT window
frame_size = int(sqrt(window))      # Size of square frames
# n_features = 64                   # size of the preprocessing window

netEpochs = 5                   # Number of epochs to train Net
netBatch = 1024                 # Number of batchs to train Net
ConvFilters = (2, 2)            # Filter for Convolution Net
ConvPool = (2, 2)               # Pool size for Convolutional Net
LSTMCells = frame_size*16       # number of LSTM Cells in Net

#------------------------------------------------------------
#Preprocessing data to feed CNN-LSTM network (RMS e Kurtosi)
#------------------------------------------------------------
choice = int(input('\nChoose a HI to pre-process your data - (1) RMS and (2) Kurtosis: '))
if choice == 1:
    print('\nCalculating RMS from each data window')
    dataRMS = np.empty((0,1))
    for i in range(0, (len(dataset)-window+1), window):
        RMS = np.sqrt((np.sum(np.square(dataset[i:window+i]))/window))
        dataRMS = np.vstack([dataRMS, RMS])
    newDataset = np.copy(dataRMS[17:])
else:
    if choice == 2:
        print('\nCalculating Kurtosis from each data window')
        dataKurtosis = np.empty((0,1))
        for i in range(0, (len(dataset)-window+1), window):
            mean_window = np.mean(dataset[i:window+i])
            std_window = np.std(dataset[i:window+i])
            KU = (np.sum(np.power(dataset[i:window+i]-mean_window, 4))/((window-1)*np.power(std_window, 4)))
            dataKurtosis = np.vstack([dataKurtosis, KU])
        newDataset = np.copy(dataKurtosis[17:])
    else:
        print('\nYou chose a wrong number')
        print('\nNo pre-processing algorithm can be used, data remains the same')
        newDataset = np.copy(dataset)


dataReshapePlot = np.reshape(newDataset, (-1,frame_size,frame_size))
dataPlot = summing_data_3D(dataReshapePlot)


#-----------------------Multidimensional PLOT------------------------

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
ax.set_title('3D Heatmap of STFT Fourier Coef. Frames of Frequency lanes')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Freq. lane Fourier Coefficient (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(770, 830)
# plt.savefig('Figures/STFT_4D.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlot, shading='auto')
plt.title('2D Heatmap - STFT Magnitude Frames')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/STFT_2D.png') # saving the plot
plt.show()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Scaling the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
datasetScaled = scaler.fit_transform(newDataset)

# Split into dataX (input frames) and dataY (output vestor)
dataXRaw = datasetScaled[:-(window)]
dataYRaw = datasetScaled[(5*window):]


# Configuring the parameters of the input data for CNN-LSTM Network 
time_steps = 5
channels = 1

#--------------------------------------------------------------------

# Reshape data as [samples, timesteps, width, height, channels]
dataY = np.reshape(dataYRaw, (-1,frame_size,frame_size,channels))

nPredict = (5*window)
data4Vec = dataXRaw.ravel()
it = np.arange(0, (len(dataXRaw)-nPredict+1), window, dtype=int)

vfunc = np.vectorize(myfunc, signature='(x),(),()->(y)')
dataRaw = vfunc(data4Vec, it, nPredict)

dataX = np.reshape(dataRaw, (-1,time_steps,frame_size,frame_size,channels))

data4prediction = np.copy(dataX)

#--------------------------CNN-LSTM Network--------------------------
# define the model
model = Sequential()
model.add(TimeDistributed(Conv2D(frame_size, ConvFilters, activation="relu"),input_shape=(None,frame_size,frame_size,channels)))
model.add(TimeDistributed(MaxPooling2D(pool_size=ConvPool)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(LSTMCells))
model.add(Dense(window, activation="linear"))
model.add(Reshape((frame_size,frame_size,channels)))
model.compile(loss='mean_squared_error', optimizer="adam")
# my_callbacks = [
#      callbacks.EarlyStopping(monitor = 'loss',
#                              min_delta = 0,
#                              patience = 3,
#                              verbose = 0,
#                              restore_best_weights = True),
#      callbacks.ModelCheckpoint(filepath = 'Models/CNN-LSTM_model.{epoch:02d}-{loss:.2f}.h5',
#                                monitor = 'loss',
#                                verbose = 0,
#                                save_best_only = True,
#                                mode = 'auto',
#                                save_freq = 1),
#      callbacks.TensorBoard(log_dir='./Logs'),
# ]
print('\n', model.summary(), end='\n')
print('\nTraining the Net...', end='\n')
model.fit(dataX, dataY, epochs=netEpochs, batch_size=netBatch, verbose=1)#, callbacks=my_callbacks)

#--------------------------------------------------------------------

# evaluate model
print('\nEvaluating the Net...', end='\n')
loss = model.evaluate(dataX, dataY, verbose=1)
print("\nTrain loss: %f" % loss, end='\n')

# make predictions
print('\nMaking predictions......', end='\n')
testPredict = model.predict(data4prediction, verbose=1)

#**************************************************************
# reshaping predict sets to return to the original scaled values
testPredictReshaped = np.reshape(testPredict, (-1,1))

# invert predictions
testPredictPlot = scaler.inverse_transform(testPredictReshaped)
#***************************************************************

# Forming new frames
dataPlotPTest = np.reshape(testPredictPlot, (-1,frame_size,frame_size))
dataPlotPTestSum = summing_data_3D(dataPlotPTest)


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
ax.set_title('3D Heatmap of TESTED STFT Amplitude Frames of Frequency lanes after prediction')
ax.set_xlabel('Frame width', labelpad=10)
ax.set_ylabel('Frame height', labelpad=10)
ax.set_zlabel('Freq. lane Amplitude (Sum)', labelpad=10)
# Set z-limit
#ax.set_zlim(0, 750)
# plt.savefig('Figures/STFT_4D.png') # saving the plot
plt.show()

plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.pcolormesh(X, Y, dataPlotPTestSum, shading='auto', cmap='plasma') #, vmin=0, vmax=0.5)
plt.title('2D Heatmap - "Test Predict"')
plt.ylabel('Frame Width')
plt.xlabel('Frame Height')
# plt.savefig('Figures/STFT_2D.png') # saving the plot
plt.show()



# #--------------------------------------------------------------------
# # prediction on new data
# print('\nPredicting on new data...', end='\n')
# yhat = model.predict(dataX[-1:,:], verbose=1)
# expected = dataX[-1:,:].reshape(-1,1)
# expected = expected[-256:,:]
# predicted = yhat.reshape(-1,1)
# trainScore = math.sqrt(mean_squared_error(expected, predicted))
# print('\nPredict Score: %.2f RMSE' % (trainScore), end='\n')


#--------------------------------------------------------------------
# Comparing the original and autorregression data
flattenDataTest = np.reshape(dataPlotPTest, (-1,1))

fig, axs = plt.subplots(2, 1, figsize=[18.00, 6.00])
fig.suptitle('Periodogram of STFT Vibration Signal and Autoregression from CNN-LSTM', fontsize=20)
axs[0].plot(dataset, label="STFT Transformed Vibration Signal X")
axs[0].set_ylabel('Amplitude', fontsize=15)
axs[0].legend(loc="upper left", fontsize=12)
axs[1].plot(flattenDataTest, 'tab:orange', label='Autoregression from CNN-LSTM Network')
axs[1].set_xlabel('STFT Timesteps - %d frequency lanes x %d steps of time' % (window, len(newDataset)), fontsize=15)
axs[1].set_ylabel('Amplitude', fontsize=15)
axs[1].legend(loc="upper left", fontsize=12)
# plt.savefig('Figures/Name_it!.png') # saving the plot
plt.show()

#--------------------------------------------------------------------
# # Finding the maximum energy of periodogram
# maxCoeff = np.amax(flattenDataTest)
# resultMaxCoeff = np.where(flattenDataTest == np.amax(flattenDataTest))
# print('\nThis is the maximum Fourier Coefficient of periodogram: %.6f' % maxCoeff)
# print('\nThis is the sample point of maximum Fourier Coefficient of periodogram: %.0f' % resultMaxCoeff[0])

#--------------------------------------------------------------
#---------------------Studing the model------------------------


# #------------------Prediction from future data-----------------
# data4stack = np.empty((0,frame_size,frame_size,1))

# indexPred = int(resultMaxCoeff[0])
# data4future = datasetScaled[(indexPred-(5*window)):indexPred]

# framedFutureData = np.reshape(data4future, (-1,time_steps,frame_size,frame_size,channels))
# iteration = 30


# for i in range(iteration):
#     print('\nFuture Prediction on data %d of %d' % ((i+1), iteration), end='\n')
#     futurePredict = model.predict(framedFutureData, verbose=1)
    
#     data4stack = np.vstack([data4stack, futurePredict])
    
#     tempFutureData = np.reshape(framedFutureData, (-1,frame_size,frame_size,channels))    
#     tempFutureData = np.vstack([tempFutureData, futurePredict])
#     tempFutureData = tempFutureData[1:]
    
#     framedFutureData = np.reshape(tempFutureData, (-1,time_steps,frame_size,frame_size,channels))


# data4plot = np.reshape(data4stack, (-1,1))

# data4plotScaled = scaler.inverse_transform(data4plot)

# fig, axs = plt.subplots(2, 1, figsize=[12.00, 6.00])
# fig.suptitle('Periodogram of Autoregression and Data Extrapolation from CNN-LSTM (%.0f times)' % iteration, fontsize=20)
# axs[0].plot(flattenDataTest[(indexPred-(5*window)):indexPred], label="Autoregression from CNN-LSTM")
# axs[0].set_ylabel('Amplitude', fontsize=15)
# axs[0].legend(loc="upper left", fontsize=12)
# axs[1].plot(data4plot, 'tab:orange', label='Extrapolation from CNN-LSTM Network')
# axs[1].set_xlabel('Time segments', fontsize=15)
# axs[1].set_ylabel('Amplitude', fontsize=15)
# axs[1].legend(loc="upper left", fontsize=12)
# # plt.savefig('Figures/Name_it!.png') # saving the plot
# plt.show()
#--------------------------------------------------------------
#--------------------------------------------------------------

#***
#***
#***
#--------------------------------------------------------------------
# Saving data into a numpy binary file
np.save('DATA/CNN-LSTM_Ber1_3_RMS.npy', flattenDataTest)
