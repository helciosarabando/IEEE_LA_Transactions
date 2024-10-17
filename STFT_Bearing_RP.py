# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:39:55 2021

@author: Hélcio Sarabando

The Short-time Fourier transform (STFT), is a Fourier-related transform 
used to determine the sinusoidal frequency and phase content of local 
sections of a signal as it changes over time.
https://en.wikipedia.org/wiki/Short-time_Fourier_transform

Compute the Short Time Fourier Transform (STFT).
STFTs can be used as a way of quantifying the change of a 
nonstationary signal’s frequency and phase content over time.

"""

print(__doc__)


# Importing libraries and solving dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
#import seaborn as sns
from matplotlib import cm

"""
Anotações:
    STFT: t = segmentos de tempo, f = frequências, 
    Zxx = vetor complexo, cujo módulo representa a amplitude de um
    elemento da série,
    nperseg = nº de bandas de frequência,
    window = tratamento da janela de frequências - hann = 1/2cos.
"""

# Parameters for an "STFT" algorithm
freq = 25.6e3       # Sampling frequency
n = 127             # Number of lanes
window = "hann"     # 1/2 cos = hanning window

def makestft (signaltotransf):
    f, t, Zxx = signal.stft(signaltotransf, fs=freq, 
                        nperseg=n,
                        noverlap=n//2,
                        window=window, 
                        return_onesided=True)
    return f, t, Zxx

#***
#***
#***
#--------------------------------------------------------------------
# Loading dataset as pandas dataframe
path = 'DATA/FEMTO_Git/Bearing1_3.csv'

dataframe = pd.read_csv(path, 
                        header=None, 
                        index_col=False)

# Preparing the data choosing only vibration signal on columns 4 (X) and 5 (Y)
signalX, signalY = dataframe.iloc[:, 4:5], dataframe.iloc[:, 5:]
signalX.rename(columns = {4:'Signal_X'}, inplace=True)
signalY.rename(columns = {5:'Signal_Y'}, inplace=True)
#--------------------------------------------------------------------

# Separating the "Healthy" part of the vibration signal in X and Y orientation
sizeH = 0.2 # amount of vibration signal considered healthy
healthX = signalX.iloc[:(int(len(signalX)*sizeH))]
healthY = signalY.iloc[:(int(len(signalY)*sizeH))]
healthX.rename(columns = {'Signal_X':'Health_Signal_X'}, inplace=True)
healthY.rename(columns = {'Signal_Y':'Health_Signal_Y'}, inplace=True)


# Ploting the vibration signal at the X orientation and the healthy amount of it
fig, (ax1, ax2) = plt.subplots(2, figsize=[18.00, 6.00])
fig.suptitle('Vibrational signals from "X" orientation', fontsize=20)
ax1.plot(signalX, label="Full vibration data")
ax1.legend(loc="upper left", fontsize=12)
ax2.plot(healthX, 'tab:orange', label="Health vibration data")
ax2.set_xlabel('Time segments', fontsize=20)
ax2.legend(loc="upper left", fontsize=12)
#plt.savefig('Figures/X_Vibration_signal.png') # saving the plot


# Ploting the vibration signal at the Y orientation and the healthy amount of it
fig, (ax1, ax2) = plt.subplots(2, figsize=[18.00, 6.00])
fig.suptitle('Vibrational signals from "Y" orientation', fontsize=20)
ax1.plot(signalY, label="Full vibration data")
ax1.legend(loc="upper left", fontsize=12)
ax2.plot(healthY, 'tab:orange', label="Health vibration data")
ax2.set_xlabel('Time segments', fontsize=20)
ax2.legend(loc="upper left", fontsize=12)
#plt.savefig('Figures/Y_Vibration_signal.png') # saving the plot
plt.show() # showing the plot
#--------------------------------------------------------------------

# Turning the vibration signal X into a numpy vector and reshaping it
# to prepare it to be fed into a STFT algorithm
Signal_X = np.array(signalX).reshape(-1,)
Signal_Y = np.array(signalY).reshape(-1,)


# Turning the healthy signal X into a numpy vector and reshaping it
# to prepare it to be fed into a STFT algorithm
Health_X = np.array(healthX).reshape(-1)
Health_Y = np.array(healthY).reshape(-1)
#*******************************************************************


# Extracting the "Short-Time Fourier Transform" from the vibration signal X
f, t, Zxx = makestft(Signal_X)

#********************************************************************
# Extracting the "Short-Time Fourier Transform" from the healthy signal X
f_HX, t_HX, Zxx_HX = makestft(Health_X)

# Extracting the "Short-Time Fourier Transform" from the healthy signal Y
f_HY, t_HY, Zxx_HY = makestft(Health_Y)

#********************************************************************

# Ploting the first window of STFT Magnitude from "transformed signal X"
plt.figure(figsize=[18.00, 6.00])
plt.bar(f, np.abs(Zxx[:,0]), width=15)
plt.title('Spectrum - STFT Coefficients - First segment with %d lanes' % ((n+1)/2), fontsize=20)
plt.ylabel('Fourier Coefficients', fontsize=20)
plt.xlabel('Frequency [Hz]', fontsize=20)
#plt.savefig('Figures/X_STFT_1stwindow.png') # saving the plot
plt.show()

# Ploting the last window of STFT Magnitude from "transformed signal X"
plt.figure(figsize=[18.00, 6.00])
plt.bar(f, np.abs(Zxx[:,-1]), width=15)
plt.title('Spectrum - STFT Coefficients - Last segment with %d lanes' % ((n+1)/2), fontsize=20)
plt.ylabel('Fourier Coefficients', fontsize=20)
plt.xlabel('Frequency [Hz]', fontsize=20)
#plt.savefig('Figures/X_STFT_lastwindow.png') # saving the plot
plt.show()

#--------------------------------------------------------------------
# # Ploting the list of frequency bands
# plt.plot(f)
# plt.title('Frequency bands')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Band position')
# #plt.close()
# plt.show()
#--------------------------------------------------------------------

#***
#***
#***
# Creating a dataframe containing the module of the complex vectors
# of transformed vibration signal X
# [frequency, short time steps]
STFT_module = np.abs(Zxx)
ZxxModule_df = pd.DataFrame(STFT_module).to_csv("DATA/FEMTO_Git/Zxx_Bearing1_3.csv")
np.save('DATA/FEMTO_Git/Zxx_Bearing1_3.npy', STFT_module)
#print (type(STFT_module))

#********************************************************************
# Creating a dataframe containing the module of the complex vectors
# of transformed healthy signal X and Y
HX_STFT_module = np.abs(Zxx_HX)
HY_STFT_module = np.abs(Zxx_HY)
#********************************************************************

# Adjusting the time vector to be ploted (ms to s)
t_test = (t*1000).astype(int)
# Adjusting the frequency vector to be ploted (to integer)
f_test = f.astype(int)

# Ploting STFT windows as 3D graph
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(t_test, f_test)
surf = ax.plot_surface(X, Y, STFT_module, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
plt.title('Spectrogram - 3-D Surface - STFT Coefficients')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time segments')
#plt.savefig('Figures/X_STFT_3D.png') # saving the plot
plt.show()

# Ploting an espectrogram of tranformed Signal X
plt.pcolormesh(t, f, np.abs(Zxx), shading='auto', vmin=0, vmax=0.5)
plt.title('Spectrogram - STFT Magnitudes')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time segments')
#plt.savefig('Figures/X_Espectrogram.png') # saving the plot
plt.show()

# # Ploting a "Heatmap" of STFT magnitude data
# p1 = sns.heatmap(STFT_module, vmin=0, vmax=0.5)
# plt.title('Heatmap - STFT Magnitude')
# plt.ylabel('Frequency bands')
# plt.xlabel('Segment time')
# plt.savefig('Figures/X_Heatmap.png') # saving the plot
# plt.show()

# Transforming the STFT transformed signal X into a 1-D array
STFT_flatten = STFT_module.flatten(order='F')

#********************************************************************
# Transforming the STFT transformed healthy X and Y into a 1-D array
HX_STFT_flatten = HX_STFT_module.flatten(order='F')
HY_STFT_flatten = HY_STFT_module.flatten(order='F')
#********************************************************************

# Ploting a complex vector Zxx as a time-dependent espectrogram.
plt.figure(figsize=[18.00, 6.00])
plt.plot(STFT_flatten)
plt.title('Periodogram', fontsize=20)
plt.ylabel('Fourier Coefficients', fontsize=20)
plt.xlabel('Freq. bands (%d lanes) x Time segments' % ((n+1)/2), fontsize=20)
#plt.savefig('Figures/Zxx_X_Espectro.png') # saving the plot
plt.show()

#***
#***
#***
#--------------------------------------------------------------------
# Turning "Zxx" complex vector into a dataframe "univariable" 
# (Time steps versus Accel X)
STFT_flatten_df = pd.DataFrame(STFT_flatten).to_csv("DATA/FEMTO_Git/STFT_flat_Ber1_3.csv")
np.save('DATA/FEMTO_Git/STFT_flat_Ber1_3.npy', STFT_flatten)

# # Turning "Zxx" complex vector into a dataframe
# Zxx_df = pd.DataFrame(Zxx).to_csv("DATA/Zxx.csv")
# np.save('DATA/Zxx.npy', Zxx)
# # Transposing and turning "Zxx" complex vector into a dataframe
# Zxx_T = np.transpose(Zxx)
# Zxx_T_df = pd.DataFrame(Zxx_T).to_csv("DATA/Zxx_T.csv")


# #********************************************************************
# # Turning "Zxx" complex vector of healthy X and Y into a dataframe "univariable" 
# # (Time steps versus Accel X)
# HX_STFT_flatten_df = pd.DataFrame(HX_STFT_flatten).to_csv("DATA/HX_STFT_flatten.csv")
# HY_STFT_flatten_df = pd.DataFrame(HY_STFT_flatten).to_csv("DATA/HY_STFT_flatten.csv")
# #-------------------------------------------------------------------
