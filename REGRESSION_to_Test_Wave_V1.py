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

import pickle

#-------------------------------------------------------------------
# Defining a function to be fitted into the data as a curve
def f_polly(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d

def f_exp(x, a, b, c):
    return a * np.exp(b * x) + c

#-----------------------------------------------------------

path = 'C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/LSTM/DATA/CNN-LSTM_Ber1_3_RMS_Wave.npy'

# Loading data from a numpy binary file
dataTest = np.load(path)

#***
#***
#***
windowTest = 64   # size of the data window in data points


# Adjusting data to be multiples of the "Window"
print('\nWait... testing data to see if matches "window" size...')
testDataWT = len(dataTest)%(windowTest)
if testDataWT != 0:
    print('\nData is not compatible with "window" size... Wait until adjusting it...')
    bestLenDataWT = int(len(dataTest)/(windowTest))
    newLenDataWT = bestLenDataWT*(windowTest)
    beginDataWT = len(dataTest)-newLenDataWT
    dataTest = dataTest[beginDataWT:]
    print('\nData is read to be reshaped!')
else:
    print('\nData length match "window" size!')



#---------------------Calculating data energy-------------------------
# Energy of the spectrum = sum of (data magnitudes)^2
# It was told by theory that the energy spectrum of a given 
# signal is the sum of the squared magnitudes.
modelTFWindowedT = np.copy(dataTest).reshape(-1,windowTest)
energyTFArrayT = np.sum(np.square(modelTFWindowedT, dtype='float64'), axis=1, dtype='float64')
energyTF2DT = np.copy(energyTFArrayT).reshape(-1,1)


#------------Defining the maximum energy of the signal---------------- 
maxEnergyTest = np.amax(energyTF2DT)
resultMaxEnergyTest = np.where(energyTF2DT == np.amax(energyTF2DT))
print('\nThis is the maximum Energy of periodogram: %.6f' % maxEnergyTest)
print('\nThis is the sample point of Energy of periodogram: %.0f' % resultMaxEnergyTest[0])

# Write the max energy found in data to a file
file = open("DATA/maxEnergyTestWave.txt", "w")
strMaxEnergyTest = repr(maxEnergyTest)
file.write("Max Energy = " + strMaxEnergyTest + "\n")

file.close()
#-----------------------------------------------------------



#------------------Calculating data moving average--------------------
# Making a Moving Average of the data to create a trend
print("\nExtracting the Moving Average of the Energy data...", end='\n')
df = pd.DataFrame(energyTF2DT, columns=['Data points'])

windowMA = windowTest*10 # Moving Average Spam

weights = np.arange(1,(windowMA+1))

# Calculating a "Window" samples span WMA using a custom function
wmaDataT = df['Data points'].rolling(windowMA).apply(lambda samples: np.dot(samples, weights)/weights.sum(), raw=True)
df['Window points WMA'] = np.round(wmaDataT, decimals=3)

# Calculating a "Window" samples span SMA
smaDataT = df['Data points'].rolling(windowMA).mean()
df['Window points SMA'] = np.round(smaDataT, decimals=3)

# Calculating a "Window" samples span EMA
emaDataT = df['Data points'].ewm(span=windowMA).mean()
df['Window points EMA'] = np.round(emaDataT, decimals=3)

# Plotting the data and the EMA, SMA and WMA of the data
plt.figure(figsize = (12,6))
plt.title('Moving Average of Energy Data Points (Test)', fontsize=18)
plt.plot(energyTF2DT, color='yellow', label="Energy of original data")
plt.plot(emaDataT, 'k--', label="%s points Exponetial Weighted Moving Average" % windowMA)
plt.plot(wmaDataT, 'b--', label="%s points Linear Weighted Moving Average" % windowMA)
plt.plot(smaDataT, 'g--', label="%s points Simple Moving Average" % windowMA)
plt.xlabel("Segments of time", fontsize=15)
plt.ylabel("Amplitude", fontsize=15)
plt.legend(fontsize=12)
plt.show()

last10Wind = int(windowTest*10)
plt.title('Exponentialy Weighted Average of Energy (Test)', fontsize=14)
plt.plot(emaDataT[-last10Wind:], label="EMA of %s samples in the last %s points" % (windowMA, last10Wind))
plt.xlabel("Segments of time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.show()



#------------------------------------------------------
# SVR way of doing it
# windowSVRT = 5000
# yT = energyTF2DT[-(windowSVRT+1):-1].ravel()
# xT = np.arange(0, len(yT)).reshape(-1,1)

# svrT = SVR(verbose=True).fit(xT, yT)
# print(svrT)

# yfitT = svrT.predict(xT)

# yPrepT = np.reshape(yT, (-1,1))

# # scalerSVR = MinMaxScaler(feature_range=(0, 1))
# # yScaled = scalerSVR.fit_transform(yPrep)

# plt.figure(figsize=(15,5))
# plt.title('SVR of Energy in the last %s points (Test)' % len(yT), fontsize=18)
# plt.scatter(xT, yPrepT, color='yellow', label='Energy Data')
# plt.plot(yfitT, 'r--', label='SVR Fitting Data')
# plt.xlabel("Segments of time", fontsize=15)
# plt.ylabel("Amplitude", fontsize=15)
# plt.legend(fontsize=12)
# plt.show()

# scoreSVRT = svrT.score(xT, yT)
# print("R-squared:", scoreSVRT)
# print("MSE:", mean_squared_error(yT, yfitT))

# #**************************************************************
# Creating a polynomial curve to fit on moving average
windowWMAFitT = 1000
yDataRawT = np.array(emaDataT)
yDataT = yDataRawT[-(windowWMAFitT+1):-1].reshape(-1,)        
xDataT = (np.arange(1, (len(yDataT)+1)).reshape(-1,))/100

#------Rescuing threshould from EMA in Learn set------
with open('threshouldEmaWave', 'rb') as fp:
    threshouldEma = pickle.load(fp)

print('\nThreshould EMA: %.6f' % threshouldEma[0])

poptExp, pcovExp = curve_fit(f_exp, xDataT, yDataT)
aT, bT, cT = poptExp
print('\nExponential fit (EMA): y = %.5f * (%.5f * exp(x)) + %.5f' % (aT, bT, cT))

poptPol, pcovPol = curve_fit(f_polly, xDataT, yDataT)
aTP, bTP, cTP, dTP = poptPol
print('\nPolly fit (EMA): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (aTP, bTP, cTP, dTP))

plt.figure(figsize=(12,6))
plt.title('Curve fit in exponentialy average energy points (Test)', fontsize=18)
plt.plot(xDataT, yDataT, 'g', label='Original Data: last %s average energy points' % len(yDataT))
#plt.plot(xDataT, np.repeat(threshouldEma, len(xDataT)), 'r--', label='Threshould from Learn set')
plt.plot(xDataT, f_exp(xDataT, *poptExp), 'b--',
          label='Exponential fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(poptExp))
plt.plot(xDataT, f_polly(xDataT, *poptPol), 'k',
          label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPol))
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
#plt.ylim(bottom=0, top=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()

#**************************************************************
# # Creating curves to fit on energy SVR Model 
# windowSVRFitT = 1000
# yDataSVRT = yfitT[-(windowSVRFitT+1):-1]
# xDataSVRT = (np.arange(1, (len(yDataSVRT)+1)).reshape(-1,))/100

# # with open('threshouldSVRWave', 'rb') as fp:
# #     threshouldSVR = pickle.load(fp)

# # print('\nThreshould SVR: %.6f' % threshouldSVR[0])

# #------Rescuing threshould from EMA in Learn set------
# with open('threshouldEmaWave', 'rb') as fp:
#     threshouldEma = pickle.load(fp)

# print('\nThreshould EMA: %.6f' % threshouldEma[0])

# poptSVRExp, pcovSVRExp = curve_fit(f_exp, xDataSVRT, yDataSVRT)
# aSVRExp, bSVRExp, cSVRExp = poptSVRExp
# print('\nExponential fit (EMA): y = %.5f * (%.5f * exp(x)) + %.5f' % (aSVRExp, bSVRExp, cSVRExp))

# poptSVRPol, pcovSVRPol = curve_fit(f_polly, xDataSVRT, yDataSVRT)
# aSVRPol, bSVRPol, cSVRPol, dSVRPol = poptSVRPol
# print('\nPolly fit (SVR): y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (aSVRPol, bSVRPol, cSVRPol, dSVRPol))

# plt.figure(figsize=(12,6))
# plt.title('Curve fit in exponentialy average energy points (Test)', fontsize=18)
# plt.plot(xDataSVRT, yDataSVRT, 'y--', label='Original Data: last %s SVR modeled energy points' % len(yDataSVRT))
# plt.plot(xDataSVRT, np.repeat(threshouldEma, len(xDataSVRT)), 'r--', label='Threshould from Learn set')
# plt.plot(xDataSVRT, f_exp(xDataSVRT, *poptSVRExp), 'b--',
#           label='Exponential fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(poptSVRExp))
# plt.plot(xDataSVRT, f_polly(xDataSVRT, *poptSVRPol), 'k',
#           label='Polly fit (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptSVRPol))
# plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
# plt.ylabel('Amplitude', fontsize=15)
# #plt.ylim(bottom=0, top=3)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.show()
#*************************************************************

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
# RUL projection using polinomial curve based in EWMA energy datapoints
xDataEMAProj = np.linspace(xDataT[-1], 12*xDataT[-1], num=len(xDataT))

# Rescuing curve coefficients from Learn set
with open('curveCoefficientsWave', 'rb') as fp:
    popt, poptPolly = pickle.load(fp)

markerEMAP = np.where(f_polly(xDataEMAProj, *poptPol) >= threshouldEma)
indMarkerEMAP = np.array(xDataEMAProj[markerEMAP[0][0]])
markerEMAE = np.where(f_exp(xDataEMAProj, *poptExp) >= threshouldEma)
indMarkerEMAE = np.array(xDataEMAProj[markerEMAE[0][0]])

Tenergy = 16e-3
Tlatency = 9.9
latencyWindowP = ((indMarkerEMAP-10)*100)/6.25
latencyWindowE = ((indMarkerEMAE-10)*100)/6.25

RUL_EMARawP = (((indMarkerEMAP-10)*100)*Tenergy)+(Tlatency*latencyWindowP)
RUL_EMARawE = (((indMarkerEMAE-10)*100)*Tenergy)+(Tlatency*latencyWindowE)
# print('\nThe Remaining Useful Life using EWMA base line is %.3f seconds' % RUL_EMARaw)


plt.figure(figsize=(12,6))
plt.title('Data extrapolation from EWMA modeled energy points (Test)', fontsize=18)
plt.plot(xDataT, yDataT, 'g', label='Original Data: last %s EWMA modeled energy points' % len(yDataT))
plt.plot(np.append(xDataT, xDataEMAProj), np.repeat(threshouldEma, (len(xDataT)+len(xDataEMAProj))), 'r--', label='Threshould from Learn set')
plt.plot(xDataEMAProj, f_polly(xDataEMAProj, *poptPol), 'y--',
          label='Polly curve (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
plt.plot(indMarkerEMAP, threshouldEma, marker='s', color='y', markersize=10, label='Point to estimate RUL from polly curve')
plt.text((indMarkerEMAP*0.9), (threshouldEma[0]*0.85), r'R.U.L.= %.2f seg.' % RUL_EMARawP, fontsize=12)
plt.plot(xDataEMAProj, f_exp(xDataEMAProj, *poptExp), 'b--',
          label='Exponential curve (coef.: a=%5.5f, b=%5.5f, c=%5.5f)' % tuple(poptExp))
plt.plot(indMarkerEMAE, threshouldEma, marker='s', color='b', markersize=10, label='Point to estimate RUL from exponential curve')
plt.text((indMarkerEMAE*0.9), (threshouldEma[0]*1.1), r'R.U.L.= %.2f seg.' % RUL_EMARawE, fontsize=12)
plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(bottom=0, top=1800)
plt.legend(fontsize=12)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
#RUL projection using polinomial curve based in SVR energy datapoints
# xDataSVRProj = np.linspace(xDataSVRT[-1], 2*xDataSVRT[-1], num=len(xDataSVRT))

# markerSVR = np.where(f_polly(xDataSVRProj, *poptPolly) >= threshouldEma)
# indMarkerSVR = np.array(xDataSVRProj[markerSVR[0][0]])

# RUL_SVRRaw = (((((indMarkerSVR-10)*100)*windowTest)+17)*64)
# RUL_SVRCalc1 = RUL_SVRRaw/2560
# RUL_SVRCalc2 = RUL_SVRCalc1*9.9
# RUL_SVRCalc3 = ((RUL_SVRRaw*39.0625e-06)/1e6)+RUL_SVRCalc2
# print('\nThe Remaining Useful Life using SVR base line is %.3f seconds' % RUL_SVRCalc3)

# plt.figure(figsize=(12,6))
# plt.title('Data extrapolation from SVR modeled energy points (Test)', fontsize=18)
# plt.plot(xDataSVRT, yDataSVRT, 'g', label='Original Data: last %s SVR modeled energy points' % len(yDataSVRT))
# plt.plot(np.append(xDataSVRT, xDataSVRProj), np.repeat(threshouldEma, (len(xDataSVRT)+len(xDataSVRProj))), 'b--', label='Threshould from Learn set')
# plt.plot(xDataSVRProj, f_polly(xDataSVRProj, *poptPolly), 'y--',
#           label='Polly curve (coef.: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f)' % tuple(poptPolly))
# plt.plot(indMarkerSVR, threshouldEma, marker='s', color='r', markersize=10, label='Point to estimate RUL')
# plt.xlabel('Energy segments over time (x10\u00b2)', fontsize=15)
# plt.ylabel('Amplitude', fontsize=15)
# #plt.ylim(bottom=0, top=3)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.show()