# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:02:24 2020

@author: Cyber Micro
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

#OneHotEncoder 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(ulke)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])



#DataFrame 
sonuc=pd.DataFrame(data=onehot_encoded,index=range(22),columns=['fr','tr','us'])

#print(sonuc)

cinsiyet=veriler.iloc[:,-1].values  
print(cinsiyet)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
sonuc5=pd.concat([sonuc,sonuc2],axis=1)
sonuc4=pd.concat([sonuc,sonuc2,sonuc3])
print(sonuc4)
print("                 ")
print(sonuc5)