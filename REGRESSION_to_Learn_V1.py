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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.optimize import curve_fit

import pickle

#-------------------------------------------------------------------
# Defining a function to be fitted into the data as a curve
def f(x, a, b, c):
    return a * np.exp(b * x) + c

def f2(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d
#-----------------------------------------------------------

path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/LSTM/DATA/CNN-LSTM_Ber1_1.npy'

# Loading data from a numpy binary file
data = np.load(path)


#***
#***
#***
window = 64   # size of the data window in data points

#-----------------------------------------------------------
# Take a mean of the amplitudes of the frequency lane in the window 
modelTFArray = data[:].reshape(-1,window)
modelTFMean = np.mean(modelTFArray, axis=1)

#-----------------------------------------------------------
# Energy of the spectrum = sum of fft_result_magnitudes^2
# It was told by theory that the energy spectrum of a given 
# signal is the sum of the squared fourier coefficient.
energyTFArray = np.sum(np.square(modelTFArray, dtype='float64'), axis=1, dtype='float64')
energyTF2D = energyTFArray[:].reshape(-1,1)

maxEnergy = np.amax(energyTF2D)
resultMaxEnergy = np.where(energyTF2D == np.amax(energyTF2D))
print('\nThis is the maximum Energy of periodogram: %.6f' % maxEnergy)
print('\nThis is the sample point of Energy of periodogram: %.0f' % resultMaxEnergy[0])

# Write the max energy found in data to a file
file = open("DATA/maxEnergyTrain.txt", "w")
strMaxEnergy = repr(maxEnergy)
file.write("Max Energy = " + strMaxEnergy + "\n")

file.close()
#-----------------------------------------------------------

scalerData = MinMaxScaler(feature_range=(0, 1))
yScaledData = scalerData.fit_transform(energyTF2D)

#-----------------------------------------------------------
# Making a Moving Average of the data to create a trend
print("\nExtracting the Moving Average of the Energy data...", end='\n')
df = pd.DataFrame(energyTF2D, columns=['Data points'])

windowMA = window*10 # Moving Average Spam

weights = np.arange(1,(windowMA+1))

# Calculating a 256 samples span WMA using a custom function
wma2560 = df['Data points'].rolling(windowMA).apply(lambda samples: np.dot(samples, weights)/weights.sum(), raw=True)
df['2560 points WMA'] = np.round(wma2560, decimals=3)

# Calculating a 256 samples span SMA
sma2560 = df['Data points'].rolling(windowMA).mean()
df['2560 points SMA'] = np.round(sma2560, decimals=3)

# Calculating a 256 samples span EMA
ema2560 = df['Data points'].ewm(span=windowMA).mean()
df['2560 points EMA'] = np.round(ema2560, decimals=3)

# Plotting the data and the SMA and WMA of the data
plt.figure(figsize = (12,6))
plt.title('Moving Average of Energy Data Points (Train)', fontsize=18)
plt.plot(yScaledData, label="Model (Energy) original data (Scaled: 0-1)")
plt.plot(ema2560, 'r--', label="%s points Exponetial Weighted Moving Average" % windowMA)
plt.plot(wma2560, 'g--', label="%s points Linear Weighted Moving Average" % windowMA)
plt.plot(sma2560, 'y--', label="%s points Simple Moving Average" % windowMA)
plt.xlabel("Segments of time", fontsize=15)
plt.ylabel("Amplitude", fontsize=15)
plt.legend(fontsize=12)
plt.show()

lastWind = 512
plt.title('Exponentialy Weighted Average of Energy (Train)', fontsize=14)
plt.plot(ema2560[-512:], label="EMA of %s samples in the last %s points" % (windowMA, lastWind))
plt.xlabel("Segments of time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.show()

#*****
# freqLane = 128
# energyPlot = np.square(modelTFArray, dtype='float64')
# plt.figure(figsize = (12,6))
# plt.title('Energy Data Points over time (Train)', fontsize=18)
# plt.plot(energyPlot[:,freqLane], label='Energy of 1 freq. lane over time - lane nº: %.0f ' % freqLane)
# plt.xlabel("Segments of time", fontsize=15)
# plt.ylabel("Amplitude", fontsize=15)
# plt.legend(fontsize=12)
# plt.show()
#*****


#------------------------------------------------------
# SVR way of doing it
windowSVR = 20000
y = energyTF2D[-(windowSVR+1):-1].ravel()
x = np.arange(0, windowSVR).reshape(-1,1)

svr = SVR(verbose=True).fit(x, y)
print(svr)

yfit = svr.predict(x)

yPrep = np.reshape(y, (-1,1))

scalerSVR = MinMaxScaler(feature_range=(0, 1))
yScaled = scalerSVR.fit_transform(yPrep)

plt.figure(figsize=(15,5))
plt.title('SVR of Energy in the last %s points (Train)' % len(y), fontsize=18)
plt.scatter(x, yScaled, label='Energy Data (Scaled: 0-1)')
plt.plot(yfit, 'r--', label='SVR Fitting Data')
plt.xlabel("Segments of time", fontsize=15)
plt.ylabel("Amplitude", fontsize=15)
plt.legend(fontsize=12)
plt.show()

scoreSVR = svr.score(x, y)
print("R-squared:", scoreSVR)
print("MSE:", mean_squared_error(y, yfit))

#**************************************************************
# Creating a exponential curve to fit on energy moving average 
windowWMAFit = 1000
yDataRaw = np.array(ema2560)
yData = yDataRaw[-(windowWMAFit+1):-1].reshape(-1,)        
xData = (np.arange(1, (len(yData)+1)).reshape(-1,))/100

threshouldEma = np.amax(yData)
resultThresIndex = np.where(yData == np.amax(yData))

with open('threshouldEma', 'wb') as fp:
    pickle.dump([threshouldEma], fp)

# with open('threshouldEma', 'rb') as fp:
#     threshouldEma = pickle.load(fp)    

print('\nThreshould will be: %.6f' % threshouldEma)
print('\nThe sample point of threshould: %.0f' % resultThresIndex[0])

popt, pcov = curve_fit(f, xData, yData)
a, b, c = popt
print('\nExponential fit (EMA): y = %.5f * (%.5f * exp(x)) + %.5f' % (a, b, c))

poptPolly, pcovPolly = curve_fit(f2, xData, yData)
ap, bp, cp, dp = poptPolly
print('\nPolly fit (EMA): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (ap, bp, cp, dp))

plt.figure(figsize=(12,6))
plt.title('Curve fit in exponentialy average energy points (Train)', fontsize=18)
plt.plot(xData, yData, 'g', label='Original Data: last %s average energy points' % len(yData))
plt.plot(xData, np.repeat(threshouldEma, len(xData)), 'b--', label='Threshould')
plt.plot(xData, f(xData, *popt), 'r--',
          label='Exponential fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(popt))
plt.plot(xData, f2(xData, *poptPolly), 'y--',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


#**************************************************************
# Creating a exponential curve to fit on energy SVR Model 
windowSVRFit = 3600
yDataSVR = yfit[-(windowSVRFit+1):-1]
xDataSVR = (np.arange(1, (len(yDataSVR)+1)).reshape(-1,))/100

threshouldSVR = np.amax(yDataSVR)
resultThresSVR = np.where(yDataSVR == np.amax(yDataSVR))

with open('threshouldSVR', 'wb') as fp:
    pickle.dump([threshouldSVR], fp)

print('\nThreshould will be: %.6f' % threshouldSVR)
print('\nThe sample point of threshould: %.0f' % resultThresSVR[0])

poptSVR, pcovSVR = curve_fit(f, xDataSVR, yDataSVR)
a, b, c = poptSVR
print('\nExponential fit (SVR): y = %.5f * (%.5f * exp(x)) + %.5f' % (a, b, c))

poptSVRPolly, pcovSVRPolly = curve_fit(f2, xData, yData)
ap, bp, cp, dp = poptSVRPolly
print('\nPolly fit (SVR): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (ap, bp, cp, dp))

plt.figure(figsize=(12,6))
plt.title('Curve fit in SVR modeled energy points (Train)', fontsize=18)
plt.plot(xDataSVR, yDataSVR, 'g', label='Original Data: last %s SVR modeled energy points' % len(yDataSVR))
plt.plot(xDataSVR, np.repeat(threshouldSVR, len(xDataSVR)), 'b--', label='Threshould')
plt.plot(xDataSVR, f(xDataSVR, *poptSVR), 'r--',
          label='Exp. fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(popt))
plt.plot(xDataSVR, f2(xDataSVR, *poptSVRPolly), 'y--',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
#plt.ylim(bottom=0, top=0.165)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
#*************************************************************