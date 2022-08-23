#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Natasha Bertrand
#CFTC Automated Machine Learning for Pay Transactions

#Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import re


# In[2]:


import sys
import os
import pandas as pd
from glob import glob
import warnings


# In[3]:


# getting excel files from Directory Desktop
path = "/home/nbertrand/Payroll_Data/PayrollByPeriod/"

# read all the files with extension .xlsx i.e. excel 
file_list = []
filenames = glob(path + "/*.xlsx")
for file in filenames: 
    file_list.append(file)
file_list


# In[4]:


a = [None]*len(file_list)
a


# In[5]:


dicts1 = {}
keys1 = range(len(file_list))
values1 = file_list
for i in keys1:
    dicts1[i] = values1[i]


# In[6]:


dicts1


# In[7]:


keys1


# In[8]:


dicts1[0]


# In[9]:


dicts1[0]


# In[10]:


dicts1[1]


# In[11]:


dicts1[2]


# In[12]:


dicts1[3]


# In[13]:


#dict(list(enumerate(values1)))


# In[14]:


for key in keys1:
  print(key)


# In[15]:


for val in values1:
    print(val)


# In[16]:


random_state = np.random.RandomState(42)
model = IsolationForest(n_estimators=1000, max_samples='auto', contamination=float(0.05),random_state=random_state)


# In[17]:


dataFrame_list = []
df_scores_list = []
df_anomaly_list = []
j = [None] * len(dicts1)

required_columns = [3,4,5,6,7,8,10]
for j in dicts1:
    df = pd.read_excel( dicts1[j] , usecols=required_columns)
    dataFrame_list.append(df)


# In[18]:


dataFrame_list


# In[19]:


warnings.filterwarnings('ignore')


# In[20]:


to_model_columns = df.columns


# In[21]:


df_scores_list = []
df_anomaly_list = []
df_frame_list = []
for frame in dataFrame_list:
    frame = frame.replace('',np.nan,regex=True)
    np.asarray(frame[["Gross_Pay"]],dtype=float)
    model.fit(frame[['Gross_Pay']])
    df_frame_list.append(frame)
    
    frame['scores']= model.decision_function(frame[['Gross_Pay']])
    df_scores_list.append(frame['scores'])
    
    frame['anomaly_check']= model.predict(frame[['Gross_Pay']])
    df_anomaly_list.append(frame['anomaly_check'])


# In[22]:


df_frame_list[0]


# In[23]:


#No loop allowed here: 

full_data_frame0 = df_frame_list[0]
anomalies0 = full_data_frame0.loc[full_data_frame0['anomaly_check']==-1]

full_data_frame1 = df_frame_list[1]
anomalies1 = full_data_frame1.loc[full_data_frame1['anomaly_check']==-1]

full_data_frame2 = df_frame_list[2]
anomalies2 = full_data_frame2.loc[full_data_frame2['anomaly_check']==-1]

full_data_frame3 = df_frame_list[3]
anomalies3 = full_data_frame3.loc[full_data_frame3['anomaly_check']==-1]


# In[24]:


anomaly_index0=list(anomalies0.index)
anomaly_index1=list(anomalies1.index)
anomaly_index2=list(anomalies2.index)
anomaly_index3=list(anomalies3.index)


# In[25]:


anomaly_length0 = len(anomalies0)
anomaly_length1 = len(anomalies1)
anomaly_length2 = len(anomalies2)
anomaly_length3 = len(anomalies3)


# In[26]:


print(anomaly_length0)
print(anomaly_length1)
print(anomaly_length2)
print(anomaly_length3)


# In[27]:


anomalies0.sort_values(by='scores',ascending=True)


# In[28]:


anomalies1.sort_values(by='scores',ascending=True)


# In[29]:


anomalies2.sort_values(by='scores',ascending=True)


# In[30]:


anomalies3.sort_values(by='scores',ascending=True)


# In[31]:


a=len(full_data_frame0)
b=len(full_data_frame1)
c=len(full_data_frame2)
d=len(full_data_frame3)


# In[32]:


contamination_check0 = (len(anomaly_index0)/len(full_data_frame0))
contamination_check1 = (len(anomaly_index1)/len(full_data_frame1))
contamination_check2 = (len(anomaly_index2)/len(full_data_frame2))
contamination_check3 = (len(anomaly_index3)/len(full_data_frame3))
print(contamination_check0) 
print(contamination_check1)
print(contamination_check2)
print(contamination_check3)


# In[33]:


plt.figure(figsize=(12,8))
plt.hist(anomalies0['scores'], bins=30)


# In[34]:


plt.figure(figsize=(12,8))
plt.hist(anomalies1['scores'], bins=30)


# In[35]:


plt.figure(figsize=(12,8))
plt.hist(anomalies2['scores'], bins=30)


# In[36]:


plt.figure(figsize=(12,8))
plt.hist(anomalies3['scores'], bins=30)

