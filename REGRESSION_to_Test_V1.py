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
def f_polly(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d

def f_exp(x, a, b, c):
    return a * np.exp(b * x) + c

#-----------------------------------------------------------

path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/LSTM/DATA/CNN-LSTM_Ber1_3_RMS.npy'

# Loading data from a numpy binary file
data = np.load(path)

#***
#***
#***
window = 64   # size of the data window in data points

#-----------------------------------------------------------
# Reshaping data to calculate signal energy 
modelTFArray = data[:].reshape(-1,window)

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
file = open("DATA/maxEnergyTest.txt", "w")
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

windowMA = int(window/3) # Moving Average Spam

weights = np.arange(1,(windowMA+1))

# Calculating a 256 samples span WMA using a custom function
wmaData = df['Data points'].rolling(windowMA).apply(lambda samples: np.dot(samples, weights)/weights.sum(), raw=True)
df['2560 points WMA'] = np.round(wmaData, decimals=3)

# Calculating a 256 samples span SMA
smaData = df['Data points'].rolling(windowMA).mean()
df['2560 points SMA'] = np.round(smaData, decimals=3)

# Calculating a 256 samples span EMA
emaData = df['Data points'].ewm(span=windowMA).mean()
df['2560 points EMA'] = np.round(emaData, decimals=3)

# Plotting the data and the SMA and WMA of the data
plt.figure(figsize = (12,6))
plt.title('Moving Average of Energy Data Points (Test)', fontsize=18)
plt.plot(yScaledData, label="Model (Energy) original data (Scaled: 0-1)")
plt.plot(emaData, 'r--', label="%s points Exponetial Weighted Moving Average" % windowMA)
plt.plot(wmaData, 'g--', label="%s points Linear Weighted Moving Average" % windowMA)
plt.plot(smaData, 'y--', label="%s points Simple Moving Average" % windowMA)
plt.xlabel("Segments of time", fontsize=15)
plt.ylabel("Amplitude", fontsize=15)
plt.legend(fontsize=12)
plt.show()

lastWind = int(window*2)
plt.title('Exponentialy Weighted Average of Energy (Test)', fontsize=14)
plt.plot(emaData[-lastWind:], label="EMA of %s samples in the last %s points" % (windowMA, lastWind))
plt.xlabel("Segments of time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.show()

# #*****
# freqLane = int(window/2)
# energyPlot = np.square(modelTFArray, dtype='float64')
# plt.figure(figsize = (12,6))
# plt.title('Energy Data Points over time (Test)', fontsize=18)
# plt.plot(energyPlot[:,freqLane], label='Energy of 1 data point over time - window: %.0f ' % freqLane)
# plt.xlabel("Segments of time", fontsize=15)
# plt.ylabel("Amplitude", fontsize=15)
# plt.legend(fontsize=12)
# plt.show()
# #*****


#------------------------------------------------------
# SVR way of doing it
windowSVR = 5000
y = energyTF2D[-(windowSVR+1):-1].ravel()
x = np.arange(0, len(y)).reshape(-1,1)

svr = SVR(verbose=True).fit(x, y)
print(svr)

yfit = svr.predict(x)

yPrep = np.reshape(y, (-1,1))

scalerSVR = MinMaxScaler(feature_range=(0, 1))
yScaled = scalerSVR.fit_transform(yPrep)

plt.figure(figsize=(15,5))
plt.title('SVR of Energy in the last %s points (Test)' % len(y), fontsize=18)
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
# Creating a polynomial curve to fit on moving average
windowWMAFit = 1000
yDataRaw = np.array(emaData)
yData = yDataRaw[-(windowWMAFit+1):-1].reshape(-1,)        
xData = (np.arange(1, (len(yData)+1)).reshape(-1,))/100

with open('threshouldEma', 'rb') as fp:
    threshouldEma = pickle.load(fp)

print('\nThreshould EMA: %.6f' % threshouldEma[0])

popt, pcov = curve_fit(f_polly, xData, yData)
a, b, c, d = popt
print('\nPolly fit (EMA): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (a, b, c, d))

plt.figure(figsize=(12,6))
plt.title('Polynomial fit in exponentialy average energy points (Test)', fontsize=18)
plt.plot(xData, yData, 'g', label='Original Data: last %s average energy points' % len(yData))
plt.plot(xData, np.repeat(threshouldEma, len(xData)), 'b--', label='Threshould')
plt.plot(xData, f_polly(xData, *popt), 'r--',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(popt))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.ylim(bottom=0, top=4.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()

#**************************************************************
# Creating a polynomial curve to fit on energy SVR Model 
windowSVRFit = 1000
yDataSVR = yfit[-(windowSVRFit+1):-1]
xDataSVR = (np.arange(1, (len(yDataSVR)+1)).reshape(-1,))/100

with open('threshouldSVR', 'rb') as fp:
    threshouldSVR = pickle.load(fp)

print('\nThreshould SVR: %.6f' % threshouldSVR[0])

poptSVR, pcovSVR = curve_fit(f_polly, xDataSVR, yDataSVR)
a, b, c, d = popt
print('\nPolly fit (SVR): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (a, b, c, d))

plt.figure(figsize=(12,6))
plt.title('Polynomial fit in SVR modeled energy points (Test)', fontsize=18)
plt.plot(xDataSVR, yDataSVR, 'g', label='Original Data: last %s SVR modeled energy points' % len(yDataSVR))
plt.plot(xDataSVR, np.repeat(threshouldSVR, len(xDataSVR)), 'b--', label='Threshould')
plt.plot(xDataSVR, f_polly(xDataSVR, *poptSVR), 'r--',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(popt))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.ylim(bottom=0, top=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
#*************************************************************

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
# RUL projection using polinomial curve based in EWMA energy datapoints
xDataEMAProj = np.linspace(xData[-1], 2*xData[-1], num=len(xData))
markerEMA = np.where(f_polly(xDataEMAProj, *popt) >= threshouldEma)
indMarkerEMA = np.array(xDataEMAProj[markerEMA[0][0]])

RUL_EMARaw = (((((indMarkerEMA-10)*100)*window)+17)*64)
RUL_EMACalc1 = RUL_EMARaw/2560
RUL_EMACalc2 = RUL_EMACalc1*9.9
RUL_EMACalc3 = ((RUL_EMARaw*39.0625e-06)/1e6)+RUL_EMACalc2
print('\nThe Remaining Useful Life using EWMA base line is %.3f seconds' % RUL_EMACalc3)


plt.figure(figsize=(12,6))
plt.title('Polynomial extrapolation in EWMA modeled energy points (Test)', fontsize=18)
plt.plot(xData, yData, 'g', label='Original Data: last %s EWMA modeled energy points' % len(yData))
plt.plot(np.append(xData, xDataEMAProj), np.repeat(threshouldEma, (len(xData)+len(xDataEMAProj))), 'b--', label='Threshould')
plt.plot(xDataEMAProj, f_polly(xDataEMAProj, *popt), 'y--',
          label='Polly curve (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(popt))
plt.plot(indMarkerEMA, threshouldEma, marker='s', color='r', markersize=10, label='Point to estimate RUL')
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.ylim(bottom=0, top=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
# RUL projection using polinomial curve based in SVR energy datapoints
xDataSVRProj = np.linspace(xDataSVR[-1], 2*xDataSVR[-1], num=len(xDataSVR))
markerSVR = np.where(f_polly(xDataSVRProj, *poptSVR) >= threshouldSVR)
indMarkerSVR = np.array(xDataSVRProj[markerSVR[0][0]])

RUL_SVRRaw = (((((indMarkerSVR-10)*100)*window)+17)*64)
RUL_SVRCalc1 = RUL_SVRRaw/2560
RUL_SVRCalc2 = RUL_SVRCalc1*9.9
RUL_SVRCalc3 = ((RUL_SVRRaw*39.0625e-06)/1e6)+RUL_SVRCalc2
print('\nThe Remaining Useful Life using SVR base line is %.3f seconds' % RUL_SVRCalc3)

plt.figure(figsize=(12,6))
plt.title('Polynomial extrapolation in SVR modeled energy points (Test)', fontsize=18)
plt.plot(xDataSVR, yDataSVR, 'g', label='Original Data: last %s SVR modeled energy points' % len(yDataSVR))
plt.plot(np.append(xDataSVR, xDataSVRProj), np.repeat(threshouldSVR, (len(xDataSVR)+len(xDataSVRProj))), 'b--', label='Threshould')
plt.plot(xDataSVRProj, f_polly(xDataSVRProj, *poptSVR), 'y--',
          label='Polly curve (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptSVR))
plt.plot(indMarkerSVR, threshouldSVR, marker='s', color='r', markersize=10, label='Point to estimate RUL')
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.ylim(bottom=0, top=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()