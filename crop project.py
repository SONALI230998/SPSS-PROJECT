#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn import metrics


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[8]:


crop=pd.read_csv("E:\\datasets\\cropdata1.csv")


# In[23]:


crop.head()


# In[24]:


crop.describe()


# In[10]:


crop.info()


# In[11]:


X=crop[["Soil","Month","State"]]


# In[12]:


y=crop[["Rice"]]


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)


# In[15]:


lm=LogisticRegression()


# In[16]:


lm.fit(X_train, y_train)


# In[17]:


prediction=lm.predict(X_test)


# In[18]:


print(metrics.accuracy_score(y_test, prediction))


# In[19]:


import tkinter as Tk
import tkinter
from tkinter import *


# In[39]:


top=Tk()
top.geometry('1600x800+0+0')
top.configure(background='orange')

variable = StringVar(top)
variable.set("1") # default value

variable = StringVar(top)
variable.set("1") # default value

tkinter.Label(top, text='Soils:', bg='orange',fg='black', font='none 12 bold').grid(row=1,column=0,sticky=W)
w = OptionMenu(top, variable,"1","2","3","4","5","6").grid(row=1, column=1,sticky=W)

tkinter.Label(top, text='Month:', bg='orange',fg='black', font='none 12 bold').grid(row=2,column=0,sticky=W)
w = OptionMenu(top, variable,"1","2","3","4","5","6").grid(row=2, column=1,sticky=W)

variable = StringVar(top)
variable.set("1") # default value
tkinter.Label(top, text='State:', bg='orange',fg='black', font='none 12 bold').grid(row=3,column=0,sticky=W)
w = OptionMenu(top, variable,"1","2","3","4","5","6").grid(row=3, column=1,sticky=W)

def click():
    
    

tkinter.Button(top, text='Submit',width=14, command=click).grid(row=5,column=1,sticky=W)

top.mainloop()


# In[ ]:




