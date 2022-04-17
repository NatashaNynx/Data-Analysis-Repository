#!/usr/bin/env python
# coding: utf-8

# Natasha M. Bertrand
# K-Clustering Algorithm on data from Paralyzed Veterans of America (PVA)

# Import relevant modules and read  the csv file: 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn.preprocessing import StandardScaler


get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('Cleaned_Data.csv')
data.head()
data


# Assign column names : 

# In[2]:


data.columns = ['HOMEOWNER',	'HIT',	'MALEVET',	'VIETVETS',	'WWIIVETS',
                    'LOCALGOV',	'STATEGOV',	'FEDGOV',	'CARDPROM',	'MAXADATE',
                    'NUMPROM',	'CARDPM12',	'NUMPRM12',	'NGIFTALL',	'CARDGIFT',	
                    'MINRAMNT',	'MINRDATE',	'MAXRAMNT',	'MAXRDATE',	'LASTGIFT',
                    'AVGGIFT',	'CONTROLN',	'HPHONE_D',	'CLUSTER2',	'CHILDREN',	
                    'AGE',	'GIFTAMNT']


# Standardize the data and then perform k-means clustering with 3 groups: 

# In[3]:


k = 3
Data1 = data[['CHILDREN','AVGGIFT']]
arrayData1 = np.array(Data1)
print(arrayData1)
print()
scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(arrayData1) 
print(scaled_data1)

kmeans1 = cluster.KMeans(n_clusters=k)
kmeans1.fit(scaled_data1)

labels = kmeans1.labels_
centroids = kmeans1.cluster_centers_


# Display results : 

# In[4]:


for i in range(k):
    # select only data observations with cluster label == i
    ds = scaled_data1[np.where(labels==i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o', markersize=3)
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=5.0)
plt.show()


# Standardize the data and then perform k-means clustering with 2 groups: 

# In[5]:


k = 2
Data1 = data[['VIETVETS','WWIIVETS',
              'GIFTAMNT']]
arrayData1 = np.array(Data1)
print(arrayData1)
print()
scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(arrayData1) 
print(scaled_data1)

kmeans1 = cluster.KMeans(n_clusters=k)
kmeans1.fit(scaled_data1)

labels = kmeans1.labels_
centroids = kmeans1.cluster_centers_


# Display results : 

# In[6]:


for i in range(k):
    # select only data observations with cluster label == i
    ds = scaled_data1[np.where(labels==i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o', markersize=3)
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=4.0)
    plt.setp(lines,mew=4.0)
plt.show()


# In[ ]:




