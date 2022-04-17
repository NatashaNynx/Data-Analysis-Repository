#!/usr/bin/env python
# coding: utf-8

# Natasha M. Bertrand

# This is a demonstration of a brief program that runs a multiple regression model giving a visual output of the coefficients for the variables. 
# Data is from Paralyzed Veterans of America (PVA), a non-profit organization. 

# I have begun by importing relevent modules read the csv file: 

# In[1]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import statsmodels.api as sm
df = pd.read_csv('Cleaned_Data.csv')


# Here I have named the independent variables for the model (X) and the dependent variables for the model (y)

# In[2]:


X = df[['CARDPROM', 'NUMPROM',	'NUMPRM12',	'NGIFTALL','CARDGIFT',	
                    'MINRAMNT',	'MINRDATE',	'MAXRAMNT',	'LASTGIFT',
                    'AVGGIFT']]
y = df['GIFTAMNT']

regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[3]:


print(regr.coef_)


# Here, I have assigned the variable names to the x axis and plot the regression coefficients on the y axis: 

# In[4]:


str_variable_list = ['CARDPROM', 'NUMPROM',	'NUMPRM12',	'NGIFTALL','CARDGIFT',	
                    'MINRAMNT',	'MINRDATE',	'MAXRAMNT',	'LASTGIFT',
                    'AVGGIFT']


# Below is the code for visual output using matplotlib

# In[5]:


str_variable_list = ['CARDPROM', 'NUMPROM', 'NUMPRM12',
'NGIFTALL','CARDGIFT',
'MINRAMNT', 'MINRDATE', 'MAXRAMNT', 'LASTGIFT',
'AVGGIFT']
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = regr.coef_
my_xticks = str_variable_list
plt.figure(figsize=(12,8))
plt.xticks(x, my_xticks)
plt.plot(x, y)
plt.show()

