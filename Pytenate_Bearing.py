# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:31:38 2021

@author: hsarabando

Este é um script contendo a concatenação de vários arquivos "csv".
"""

from os import listdir, chdir
from os.path import isfile, join
import matplotlib.pyplot as plt

import pandas as pd

D_path = "C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/DATA/FEMTOBearingDataSet - Git/Test_set/Bearing3_3"
path = "C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/DATA/FEMTOBearingDataSet - Git/Test_set"

chdir(D_path)
#onlyfiles = [f for f in listdir(D_path) if isfile(join(D_path, f))]

# isFile = isfile(D_path)
# print(isFile)

onlyfiles = pd.DataFrame([])
for f in listdir(D_path):
      if f[0] == "a" and isfile(join(D_path, f)):
          f = pd.DataFrame([f])
          onlyfiles = pd.concat([onlyfiles, f])
        
onlyfiles = list(onlyfiles[0])

print(onlyfiles)

#x = pd.read_csv("C:/Users/hsara/Downloads/RESEARCH/Working Algorithms/DATA/acc_00001.csv", header=None)

combined_csv = pd.concat([pd.read_csv(f, header=None) for f in onlyfiles], axis=0, ignore_index=True)

chdir(path)

combined_csv.to_csv("Bearing3_3.csv", header=False, index=False, encoding='utf-8-sig')


plt.plot(combined_csv[5])