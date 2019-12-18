#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score 


# In[8]:


data=pd.read_csv(r"C:\Users\Administrator\Documents\python\winequality-white.csv",sep=";")
print(data)


# In[11]:


data.head()


# In[13]:


data.info()


# In[19]:


data.describe()


# In[20]:


pd.isnull(data)


# 

# In[36]:


#Creating Histogram.
fig, ax = plt.subplots(1, 2) 
ax[1].hist(data.alcohol, 10, facecolor ='blue', 
           ec ="black", lw = 0.5, alpha = 0.5, 
           label ="White wine") 
fig.subplots_adjust(left = 0, right = 1, bottom = 0,  
               top = 0.5, hspace = 0.05, wspace = 1) 
ax[1].set_ylim([0, 1000]) 
ax[1].set_xlabel("Alcohol in % Vol") 
ax[1].set_ylabel("Frequency")
fig.suptitle("Distribution of Alcohol in % Vol") 
plt.show()


# In[37]:


X = data.drop('quality', axis=1)
y = data.quality


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)


# In[42]:


print(X_train.head())


# In[45]:


X_train_scaled = preprocessing.scale(X_train)
print(X_train_scaled)


# In[53]:


clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[54]:


confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)


# In[55]:


y_pred = clf.predict(X_test)


# In[ ]:




