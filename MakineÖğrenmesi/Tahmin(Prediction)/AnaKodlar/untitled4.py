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
veriler = pd.read_csv('veriler.csv')



#Veri Önİşleme

boy = veriler['boy']


boykilo=veriler[['boy','kilo']]

print(boykilo)

x=10
