# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:19:17 2020

@author: Cyber Micro
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

veriler=pd.read_csv('maaslar.csv')

X = veriler.iloc[:, 1].values
y = veriler.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X.reshape(-1,1),y)


X_grid = np.arange(min(X),max(X),0.01)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.show()




from sklearn.metrics import r2_score

r2_score(X,regressor(X_grid))