# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#verilerin olceklenmesi





from sklearn.ensemble import RandomForestRegressor


rfReg= RandomForestRegressor(random_state=0)

rfReg.fit(X,Y)

print(rfReg.predict(6.5))

