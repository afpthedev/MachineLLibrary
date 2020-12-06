

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


veriler=pd.read_csv('veriler.csv')

X=veriler.iloc[:,1:4].values
Y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
Xtrain=sc.fit_transform(x_train)
Xtest =sc.transform(x_test)


from sklearn.linear_model import LogisticRegression

LG=LogisticRegression(random_state=0)
LG.fit(Xtrain,y_train)

y_pred= LG.predict(Xtest)
print(y_pred)

#Başarı okuma Karmaşıklık Matrisi  
from sklearn.metrics import confusion_matrix
cfma=confusion_matrix(y_test,y_pred)
print(cfma)


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(metric='minkowski',n_neighbors=5)
knn.fit(Xtrain,y_train)


y_pred= knn.predict(Xtest)

cfma= confusion_matrix(y_test,y_pred)

print(cfma)


