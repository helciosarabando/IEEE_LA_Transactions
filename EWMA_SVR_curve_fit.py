# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:34:18 2021

@author: Hélcio Sarabando
Código-fonte criado a partir da pesquisa sobre vários métodos de 
pós-processamento de séries temporais
"""

# Importing libraries and solving dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler#, scale
#from sklearn.linear_model import SGDRegressor
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.optimize import curve_fit

#-------------------------------------------------------------------
# Defining a function to be fitted into the data as a curve
def f(x, a, b, c):
    return a * np.exp(b * x) + c

#-----------------------------------------------------------

path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/LSTM/DATA/CNN-LSTM_Ber1_1.npy'

# Loading data from a numpy binary file
data = np.load(path)

#-----------------------------------------------------------
# Take a mean of the amplitudes of the frequency lane in the window 
modelTFArray = data[:].reshape(-1,256)
#modelTFMean = np.mean(modelTFArray, axis=1)

# Energy of the spectrum = sum of fft_result_magnitudes^2
# It was told by theory that the energy spectrum of a given 
# signal is the sum of the squared fourier coefficient.
#-----------------------------------------------------------
energyTFArray = np.sum(np.square(modelTFArray, dtype='float64'), axis=1, dtype='float64')
energyTF2D = energyTFArray[:].reshape(-1,1)

maxEnergy = np.amax(energyTF2D)
resultMaxEnergy = np.where(energyTF2D == np.amax(energyTF2D))
print('\nThis is the maximum Energy of periodogram: %.6f' % maxEnergy)
print('\nThis is the sample point of Energy of periodogram: %.0f' % resultMaxEnergy[0])

file = open("DATA/maxEnergy.txt", "w")
strMaxEnergy = repr(maxEnergy)
file.write("Max Energy = " + strMaxEnergy + "\n")

file.close()

#-----------------------------------------------------------
#-----------------------------------------------------------

#-----------------------------------------------------------
# Making a Moving Average of the data to create a trend
df = pd.DataFrame(energyTF2D, columns=['Data points'])

weights = np.arange(1,2561)

# Calculating a 256 samples span WMA using a custom function
wma2560 = df['Data points'].rolling(2560).apply(lambda samples: np.dot(samples, weights)/weights.sum(), raw=True)
df['2560 points WMA'] = np.round(wma2560, decimals=3)

# Calculating a 256 samples span SMA
sma2560 = df['Data points'].rolling(2560).mean()
df['2560 points SMA'] = np.round(sma2560, decimals=3)

# Calculating a 256 samples span EMA
ema2560 = df['Data points'].ewm(span=2560).mean()
df['2560 points EMA'] = np.round(ema2560, decimals=3)

# Plotting the data and the SMA and WMA of the data
plt.figure(figsize = (12,6))
#plt.plot(df['samples'], label="Model original data")
plt.plot(ema2560, 'r--', label="2560 points Exponetial Weighted Moving Average")
plt.plot(wma2560, 'g--', label="2560 points Linear Weighted Moving Average")
plt.plot(sma2560, 'y--', label="2560 points Simple Moving Average")
plt.xlabel("Segments of time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


plt.plot(ema2560[-512:], label="2560 samples EMA")
plt.xlabel("Segments of time")
plt.ylabel("Fourier Coefficients")
plt.legend()
plt.show()

#*****
freqLane = 128
energyPlot = np.square(modelTFArray, dtype='float64')
plt.figure(figsize = (12,6))
plt.plot(energyPlot[:,freqLane], label='Energy of 1 freq. lane over time')
plt.xlabel("Segments of time")
plt.ylabel("Fourier Coefficient")
plt.legend()
plt.show()
#*****

# #-----------------------------------------------------------
# # Data regression to create a trend (exponential function?!?)
# # SGD Regression way of doing it
# ySGD = energyTF2D[1:].ravel()
# xSGD = energyTF2D[0:-1,:]

# # To improve the model accuracy we'll scale both x and y data 
# # and split them
# xScaled = scale(xSGD)
# yScaled = scale(ySGD)
# xtrain, xtest, ytrain, ytest = train_test_split(xSGD, ySGD, test_size=.30)

# # Define the regressor model by using the SGDRegressor class
# sgdr = SGDRegressor(loss = 'squared_loss', penalty = 'l2')
# print(sgdr)

# # Fitting the model on train data and check the model accuracy score
# sgdr.fit(xtrain, ytrain)

# score = sgdr.score(xtrain, ytrain)
# print("R-squared:", score)

# # Predicting and accuracy check
# ypred = sgdr.predict(xtest)

# mse = mean_squared_error(ytest, ypred)
# print("MSE: ", mse)
# print("RMSE: ", mse*(1/2.0))

# # Plotting the original and predicted data
# x_ax = range(len(ytest))
# plt.figure(figsize=(12,6))
# plt.plot(x_ax, ytest, linewidth=1, label="original")
# plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
# plt.title("y-test and y-predicted data")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend(loc='best',fancybox=True, shadow=True)
# plt.grid(True)
# plt.show()

#------------------------------------------------------
# SVR way of doing it
y = energyTF2D[-20001:-1].ravel()
x = np.arange(0, 20000).reshape(-1,1)

svr = SVR(verbose=True).fit(x, y)
print(svr)

yfit = svr.predict(x)

yPrep = np.reshape(y, (-1,1))

scalerSVR = MinMaxScaler(feature_range=(0, 1))
yScaled = scalerSVR.fit_transform(yPrep)

plt.figure(figsize=(15,5))
plt.scatter(x, yScaled, label='Data')
plt.plot(yfit, 'r--', label='SVR Fitting Data')
plt.legend()
plt.show()

scoreSVR = svr.score(x, y)
print("R-squared:", scoreSVR)
print("MSE:", mean_squared_error(y, yfit))

#**************************************************************
# Creating a exponential curve to fit fault progression
yDataRaw = np.array(ema2560)
yData = yDataRaw[-1000:-1].reshape(-1,)        
xData = (np.arange(1, (len(yData)+1)).reshape(-1,))/100

popt, pcov = curve_fit(f, xData, yData)
print(popt)

plt.figure(figsize=(10,6))
plt.plot(xData, yData, 'g', label='Data')
plt.plot(xData, np.repeat(2.0, len(xData)), 'b--', label='Threshould')
plt.plot(xData, f(xData, *popt), 'r--',
          label='fit: a=%5.5f, b=%5.5f, c=%5.5f' % tuple(popt))
plt.xlabel('Energy segments over time ', fontsize=15)
plt.ylabel('Fourier Coefficients', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
#*************************************************************