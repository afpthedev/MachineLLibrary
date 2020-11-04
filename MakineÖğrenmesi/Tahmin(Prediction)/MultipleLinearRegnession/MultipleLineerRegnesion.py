# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:02:24 2020

@author: Cyber Micro
Unutulmamalıdır Ki bunların arasında verilerin test için print dosyaları bulunmaktadır
"""

#Kutuphaneler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



#Kod Satıları
#Veri Yükleme
veriler = pd.read_csv('veriler.csv')


print('             ')
print(veriler)
#Kategorik->>> Numerik 
c= veriler.iloc[:,-1:].values
print(c)

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(c[:,-1])


#OneHotEncoder 

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(c)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
c = onehot_encoder.fit_transform(integer_encoded)

print(c)


sonuc2=pd.DataFrame(data=c[:,-1],index=range(22),columns=['boy','kilo','yas'])



'''
#Data bolunmesı 2/3 oranında dikkat et
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satis,test_size=0.33,random_state=0)
'''

'''
#Data birleştirilmesi 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#standartlaşma için kullanılır 
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
'''
'''
#↨Model İnşası (Linear Regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
#fit bir model oluşturulmasını sağlar
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

#Veri Görselleştirme
x_train=x_train.sort_index()
y_train=y_train.sort_index()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title('Aylara Gore Satış')
plt.xlabel('Aylar')
plt.ylabel('Satislar')
 '''

 
 


