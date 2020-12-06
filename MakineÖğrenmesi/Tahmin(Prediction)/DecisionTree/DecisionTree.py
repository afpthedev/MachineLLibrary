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


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)


plt.scatter(X,Y)
plt.plot(X,dtr.predict(X))
plt.show()
print()


from sklearn.ensemble import RandomForestRegressor


rfReg= RandomForestRegressor(n_estimator =10 ,random_state=0)

rfReg.fit(X,Y)

rfreg.predict(6.5)

