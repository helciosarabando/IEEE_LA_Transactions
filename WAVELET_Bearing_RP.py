# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:12:55 2021

@author: Hélcio Sarabando
Código-fonte criado a partir do exemplo disponível em:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html#scipy.signal.cwt
    
"""
# Importing libraries and solving dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def makeWaveTransf (data, width, wavelet=signal.ricker):
    widths = np.arange(1, width+1)
    cwtmatr = signal.cwt(data, wavelet, widths)
    cwtmatrLastScal = cwtmatr[-1, :]
    cwtmatrFlat = cwtmatr.flatten(order='F')
    return cwtmatr, cwtmatrFlat, cwtmatrLastScal

# *******
# *******
# Loading dataset as pandas dataframe
path = 'DATA/FEMTO_Git/Bearing1_3.csv'

dataframe = pd.read_csv(path, 
                        header=None, 
                        index_col=False)

# Preparing the data choosing only vibration signal on columns 4 (X) and 5 (Y)
signalX, signalY = dataframe.iloc[:, 4:5], dataframe.iloc[:, 5:]
signalX.rename(columns = {4:'Signal_X'}, inplace=True)
signalY.rename(columns = {5:'Signal_Y'}, inplace=True)

# Ploting the vibration signal on X and Y orientation
fig, (ax1, ax2) = plt.subplots(2, figsize=[18.00, 6.00])
fig.suptitle('Vibrational signals', fontsize=20)
ax1.plot(signalX, label="X oriented vibration data")
ax1.legend(loc="upper left", fontsize=12)
ax2.plot(signalY, 'tab:green', label="Y oriented vibration data")
ax2.set_xlabel('Time segments', fontsize=20)
ax2.legend(loc="upper left", fontsize=12)
# plt.savefig('Figures/Wavelet/X_Y_Vibration_B1_3.png') # saving the plot
plt.show()

#----------------------------------------------------------------
#----------------------------------------------------------------

# Preparing the data to be fed into a Wavelet Transform algorithm
signal_X = np.array(signalX).reshape(-1,)

#============================================
# Parameters for Wavelet transform
scales = 5
#============================================

# Extracting a Wavelet Transform from the data
signalTransf, signalTFlat, signalTLastScal = makeWaveTransf(signal_X, scales)


# *******
# *******
# Saving data into "npy" numpy file
# np.save('DATA/FEMTO_Git/WAVELET_Bearing1_3.npy', signalTransf)
# np.save('DATA/FEMTO_Git/WAVELET_flat_Ber1_3.npy', signalTFlat)
np.save('DATA/FEMTO_Git/WAVELET_absflat_Ber1_3.npy', np.abs(signalTFlat))
np.save('DATA/FEMTO_Git/WAVELET_abslastscal_Ber1_3.npy', np.abs(signalTLastScal))


# Preparing data to plot
signalTransfPlot = np.array(signalTransf, dtype='float16')
signalTFlatPlot = np.array(signalTFlat, dtype='float16')
signalTLastScalPlot = np.array(signalTFlat, dtype='float16')


# Ploting an scalogram of transformed Signal X
plt.imshow(np.abs(signalTransf), cmap='viridis', aspect='auto', vmax=np.max(np.abs(signalTransf))/20)
# plt.pcolormesh(t, f, np.abs(signalTransf), shading='auto', vmin=0, vmax=0.5)
plt.title('Scaleogram')
plt.ylabel('Scales')
plt.xlabel('Time segments')
# plt.savefig('Figures/Wavelet/X_Scaleogram_B1_3.png') # saving the plot
plt.show()

#____________________________________________________________________
#____________________________________________________________________

# Ploting a flat vector as a time-dependent scalogram.
fig, (ax3, ax4) = plt.subplots(2, figsize=[18.00, 6.00])
fig.suptitle('Time series of Ricker Wavelet transformed signal', fontsize=20)
ax3.plot(signalTLastScalPlot, color='black', label="Flat concat. vibration data transformed")
ax3.set_ylabel('Scales Coef.', fontsize=14)
ax3.legend(loc="upper left", fontsize=12)
ax4.plot(np.abs(signalTLastScalPlot), color='orange', label="Flat P2P vibration data transformed")
ax4.set_xlabel('Time segments', fontsize=20)
ax4.set_ylabel('Scales Coef. (abs)', fontsize=14)
ax4.legend(loc="upper left", fontsize=12)
# plt.savefig('Figures/Wavelet/Time_Series_Scaleogram_B1_3.png') # saving the plot
plt.show()


# Plot each scale from transformed signal
fig, axiss = plt.subplots(nrows=scales, ncols=1, figsize=(12,(scales*1.5)))
for level in range(scales):
    if level==0:
        fig.suptitle('Time series of each scale from Ricker Wavelet transformed signal', fontsize=20)
    #axiss[level].set_title('Level {}'.format(level +1), color='red', fontsize=10)
    axiss[level].plot(signalTransfPlot[level, :], color='blue')
    axiss[level].set_ylabel('Level {}'.format(level +1), color='red', fontsize=15)
axiss[level].set_xlabel('Time segments', fontsize=15)
plt.tight_layout()
# plt.savefig('Figures/Wavelet/Scales_Scaleogram_B1_3.png') # saving the plot
plt.show()