#!/usr/bin/env python
# coding: utf-8

# Natasha M. Bertrand
# Example Lasso machine learning Method for Stepwise Regression
# Data provided by Paralyzed Veterans of America (PVA)

# In[1]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
df = pd.read_csv('Cleaned_Data.csv')


# #define predictor and response variables

# In[2]:


X = df[['CARDPROM', 'NUMPROM',	'NUMPRM12',	'NGIFTALL','CARDGIFT',	
                    'MINRAMNT',	'MINRDATE',	'MAXRAMNT',	'LASTGIFT',
                    'AVGGIFT']]
y = df['GIFTAMNT']


# In[3]:


# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X, y)
LassoCV(cv=5, max_iter=10000, random_state=0)
model.alpha_


# In[4]:


lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X, y)
Lasso(alpha=0.10511896611445351)
print(list(zip(lasso_best.coef_, X)))


# In[5]:


alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha')

