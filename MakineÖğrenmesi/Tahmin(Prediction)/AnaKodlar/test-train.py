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
veriler = pd.read_csv('eksikveriler.csv')



#Veri Önişleme

boy = veriler['boy']


boykilo=veriler[['boy','kilo']]



x=10
#Eksik Veriler Tamamlama Algoritması 
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Yas= veriler.iloc[:,1:4].values
#print(Yas)

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
#print(Yas)

#Kategorik Veriler Algoritması 
#LabelEncoder
ulke=veriler.iloc[:,0:1].values
#print(ulke)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

ulke[:,0]=le.fit_transform(ulke[:,0])

#print(ulke)

#OneHotEncoder İnternetten alındı
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(ulke)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])



#DataFrame ve dağılım sağlanması 
sonuc=pd.DataFrame(data=onehot_encoded,index=range(22),columns=['fr','tr','us'])

print(sonuc)

cinsiyet=veriler.iloc[:,-1].values  
print(cinsiyet)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
sonuc5=pd.concat([sonuc,sonuc2],axis=1)
sonuc4=pd.concat([sonuc,sonuc2,sonuc3],axis=1)
print(sonuc4)
print("                ")
print(sonuc5)
#Data bolunmesı 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonuc5,sonuc3,test_size=0.33,random_state=0)



#Data birleştirilmesi ve tahmin yapılması 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


 




