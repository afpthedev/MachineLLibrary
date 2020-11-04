import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler=pd.read_csv('maaslar.csv')

x=veriler.iloc[:,:1]
y=veriler.iloc[:,2:]
z=veriler.iloc[:,1:2]


from sklearn.linear_model import LinearRegression

lg=LinearRegression()
lg.fit(y.values,z.values)

plt.scatter(y.values,z.values)# Scater Gorselleştirmeye yarıyor
regpre=lg.predict(y)
plt.plot(y,regpre)


