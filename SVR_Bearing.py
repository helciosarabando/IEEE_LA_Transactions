# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:38:13 2021

@author: SARABANDO, H

Cófigo-fonte extraído do GitHub de "colinberan" no endereço:
https://github.com/colinberan/Support-Vector-Regression-in-Python/blob/master/svr.py
    
"""

# Support Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import signal

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
# dataset = pd.read_csv('DATA/Position_Salaries.csv')
# X = dataset.iloc[:, 1:2].to_numpy()
# y = dataset.iloc[:, 2].to_numpy()

dataset = pd.read_csv('DATA/combined_csv.csv')
#X = dataset.iloc[:, 0:1].to_numpy()
X = np.array(dataset.index.values.tolist())
#X = signal.resample(X, len(X)//10)
X = X[:: 500]
X = X.reshape(-1,1)

y = dataset.iloc[:, 4].to_numpy()
#y = signal.resample(y, len(y)//10)
y = y[:: 500]

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X, y = None)

# Transform y into 2D array
y = np.array(y).reshape((len(y), 1))
y = sc_y.fit_transform(y, y = None)

# Fitting SVR to the dataset
regressor = SVR(kernel = 'rbf', 
                verbose = True)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

# Invert y_pred result
y_pred = sc_y.inverse_transform(y_pred)

# # Visualising the SVR results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Position vs Salary (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))

#plt.scatter(X, y, color = 'red')
plt.plot(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), linewidth = 5, color = 'blue')
plt.title('Sensor 1 vs Cycle (SVR)')
plt.xlabel('Cycle')
plt.ylabel('Sensor 1')
plt.show()