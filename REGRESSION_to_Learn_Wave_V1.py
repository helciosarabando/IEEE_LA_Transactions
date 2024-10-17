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

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.svm import SVR
from scipy.optimize import curve_fit
#from scipy.signal import hilbert

import pickle

#-------------------------------------------------------------------
# Defining a function to be fitted into the data as a curve
# Exponential
def f(x, a, b, c):
    return a * np.exp(b * x) + c

# Polinomial
def f2(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d
#-----------------------------------------------------------

path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/LSTM/DATA/CNN-LSTM_Ber1_1_Wave.npy'

# Loading data from a numpy binary file
data = np.load(path)

#***
#***
#***
window = 64   # size of the data window in data points


# Adjusting data to be multiples of the "Window"
print('\nWait... testing data to see if matches "window" size...')
testDataW = len(data)%(window)
if testDataW != 0:
    print('\nData is not compatible with "window" size... Wait until adjusting it...')
    bestLenDataW = int(len(data)/(window))
    newLenDataW = bestLenDataW*(window)
    beginDataW = len(data)-newLenDataW
    data = data[beginDataW:]
    print('\nData is read to be reshaped!')
else:
    print('\nData length match "window" size!')
    


#---------------------Calculating data energy-------------------------
# Energy of the spectrum = sum of (data magnitudes)^2
# It was told by theory that the energy spectrum of a given 
# signal is the sum of the squared magnitudes.
modelTFWindowed = np.copy(data).reshape(-1,window)
energyTFArray = np.sum(np.square(modelTFWindowed, dtype='float64'), axis=1, dtype='float64')
energyTF2D = np.copy(energyTFArray).reshape(-1,1)

#--------------Extrating Hilbert transform from data-----------------
# analytic_signal = hilbert(energyTF2D)
# amplitude_envelope = np.abs(analytic_signal)
# energyTF2D = np.copy(amplitude_envelope)

#------------------Calculating data moving average--------------------
# Making a Moving Average of the data to create a trend
print("\nExtracting the Moving Average of the Energy data...", end='\n')
df = pd.DataFrame(energyTF2D, columns=['Data points'])

windowMA = window*10 # Moving Average Spam

weights = np.arange(1,(windowMA+1))

# Calculating a "Window" samples span WMA using a custom function
wmaWindow = df['Data points'].rolling(windowMA).apply(lambda samples: np.dot(samples, weights)/weights.sum(), raw=True)
df['Window points WMA'] = np.round(wmaWindow, decimals=3)

# Calculating a "Window" samples span SMA
smaWindow = df['Data points'].rolling(windowMA).mean()
df['Window points SMA'] = np.round(smaWindow, decimals=3)

# Calculating a "Window" samples span EMA
emaWindow = df['Data points'].ewm(span=windowMA).mean()
df['Window points EMA'] = np.round(emaWindow, decimals=3)

#--------------------------------------------------------------------
# Extracting the maximum energy magnitude from "moving average" data
maxEnergyLearn = np.amax(emaWindow)
resultMaxEnergyLearn = np.where(emaWindow == np.amax(emaWindow))
print('\nThis is the maximum Energy of scaleogram: %.6f' % maxEnergyLearn)
print('\nThis is the sample point of Energy of scaleogram: %.0f' % resultMaxEnergyLearn[0])

# Write the max energy found in data to a file
file = open("DATA/maxEnergyTrainWave.txt", "w")
strMaxEnergyLearn = repr(maxEnergyLearn)
file.write("Max Energy = " + strMaxEnergyLearn + "\n")

file.close()
#---------------------------------------------------------------------

# Plotting the data and the EMA, SMA and WMA of the data
plt.figure(figsize = (12,6))
plt.title('Moving Average of Energy Data Points (Train)', fontsize=18)
plt.plot(energyTF2D, color='yellow', label="Energy of original data")
plt.plot(emaWindow, 'k--', label="%s points Exponetial Weighted Moving Average" % windowMA)
plt.plot(wmaWindow, 'b--', label="%s points Linear Weighted Moving Average" % windowMA)
plt.plot(smaWindow, 'g--', label="%s points Simple Moving Average" % windowMA)
plt.xlabel("Segments of time", fontsize=15)
plt.ylabel("Amplitude", fontsize=15)
plt.ylim(0, 15000)
plt.legend(fontsize=12)
plt.show()

last10Wind = int(window*10)
plt.title('Exponentialy Weighted Average of Energy (Train)', fontsize=14)
plt.plot(emaWindow[-last10Wind:], label="EMA of %s samples in the last %s points" % (windowMA, last10Wind))
plt.xlabel("Segments of time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.show()




# #------------------------------------------------------
# # SVR way of doing it
# windowSVR = 20000
# y = energyTF2D[-(windowSVR+1):-1].ravel()
# x = np.arange(0, windowSVR).reshape(-1,1)

# svr = SVR(verbose=True).fit(x, y)
# print(svr)

# yfit = svr.predict(x)

# yPrep = np.reshape(y, (-1,1))

# toScale = 1000
# scalerSVR = MinMaxScaler(feature_range=(0, toScale))
# yScaled = scalerSVR.fit_transform(yPrep)

# plt.figure(figsize=(15,5))
# plt.title('SVR of Energy in the last %s points (Train)' % len(y), fontsize=18)
# plt.scatter(x, yScaled, color='yellow', label='Energy Data (scaled 0~%s)' % toScale)
# plt.plot(yfit, 'r--', label='SVR Fitting Data')
# plt.xlabel("Segments of time", fontsize=15)
# plt.ylabel("Amplitude", fontsize=15)
# plt.legend(fontsize=12)
# plt.show()

# scoreSVR = svr.score(x, y)
# print("R-squared:", scoreSVR)
# print("MSE:", mean_squared_error(y, yfit))


#**************************************************************
# Creating curves to fit on energy moving average 
windowWMAFit = 3500
yDataRaw = np.array(emaWindow)
yData = yDataRaw[-(windowWMAFit+1):-1].reshape(-1,)        
xData = (np.arange(1, (len(yData)+1)).reshape(-1,))/100

threshouldEma = np.amax(yData)
resultThresIndex = np.where(yData == np.amax(yData))

#------Saving threshould from EMA------
with open('threshouldEmaWave', 'wb') as fp:
    pickle.dump([threshouldEma], fp)

# with open('threshouldEma', 'rb') as fp:
#     threshouldEmaWave = pickle.load(fp)    

print('\nThreshould will be: %.6f' % threshouldEma)
print('\nThe sample point of threshould: %.0f' % resultThresIndex[0])

popt, pcov = curve_fit(f, xData, yData)
a, b, c = popt
print('\nExponential fit (EMA): y = %.5f * (%.5f * exp(x)) + %.5f' % (a, b, c))

poptPolly, pcovPolly = curve_fit(f2, xData, yData)
ap, bp, cp, dp = poptPolly
print('\nPolly fit (EMA): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (ap, bp, cp, dp))

#------Saving curves coefficients------
with open('curveCoefficientsWave', 'wb') as fp:
    pickle.dump([popt, poptPolly], fp)
    
# with open('curveCoefficientsWave', 'rb') as fp:
#     at, bt, ct, apt, bpt, cpt, dpt = pickle.load(fp)


plt.figure(figsize=(12,6))
plt.title('Curve fit in exponentialy average energy points (Train)', fontsize=18)
plt.plot(xData, yData, 'y--', label='Original Data: last %s average energy points' % len(yData))
plt.plot(xData, np.repeat(threshouldEma, len(xData)), 'r--', label='Threshould')
plt.plot(xData, f(xData, *popt), 'b--',
          label='Exponential fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(popt))
plt.plot(xData, f2(xData, *poptPolly), 'k',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


#**************************************************************
# # Creating a exponential curve to fit on energy SVR Model 
# windowSVRFit = 2000
# yDataSVR = yfit[-(windowSVRFit+1):-1]
# xDataSVR = (np.arange(1, (len(yDataSVR)+1)).reshape(-1,))/100

# threshouldSVR = np.amax(yDataSVR)
# resultThresSVR = np.where(yDataSVR == np.amax(yDataSVR))

# with open('threshouldSVRWave', 'wb') as fp:
#     pickle.dump([threshouldSVR], fp)

# print('\nThreshould will be: %.6f' % threshouldSVR)
# print('\nThe sample point of threshould: %.0f' % resultThresSVR[0])

# poptSVR, pcovSVR = curve_fit(f, xDataSVR, yDataSVR)
# a, b, c = poptSVR
# print('\nExponential fit (SVR): y = %.5f * (%.5f * exp(x)) + %.5f' % (a, b, c))

# poptSVRPolly, pcovSVRPolly = curve_fit(f2, xData, yData)
# ap, bp, cp, dp = poptSVRPolly
# print('\nPolly fit (SVR): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (ap, bp, cp, dp))

# plt.figure(figsize=(12,6))
# plt.title('Curve fit in SVR modeled energy points (Train)', fontsize=18)
# plt.plot(xDataSVR, yDataSVR, 'g', label='Original Data: last %s SVR modeled energy points' % len(yDataSVR))
# plt.plot(xDataSVR, np.repeat(threshouldSVR, len(xDataSVR)), 'b--', label='Threshould')
# plt.plot(xDataSVR, f(xDataSVR, *poptSVR), 'r--',
#           label='Exp. fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(popt))
# plt.plot(xDataSVR, f2(xDataSVR, *poptSVRPolly), 'y--',
#           label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
# plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
# plt.ylabel('Amplitude', fontsize=15)
# #plt.ylim(bottom=0, top=0.165)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.show()
#*************************************************************