#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score 


# In[4]:


data=pd.read_csv(r"C:\Users\Administrator\Documents\python\winequality-white.csv",sep=";")
print(data)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[9]:


pd.isnull(data)


# In[10]:


quality = data["quality"].values
category = []
for num in quality:
    if num<5:
        category.append("Bad")
    elif num>6:
        category.append("Good")
    else:
        category.append("Mid")


# In[13]:


plt.figure(figsize=(10,6))
sns.countplot(data["quality"],palette="muted")
data["quality"].value_counts()


# In[15]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True)


# In[17]:


plt.figure(figsize=(12,6))
sns.barplot(x=data["quality"],y=data["alcohol"],palette="Reds")


# In[18]:


plt.figure(figsize=(12,6))
sns.jointplot(y=data["density"],x=data["alcohol"],kind="hex")


# In[19]:


X= data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[21]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y)


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[24]:


#svm
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred_svc =svc.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_svc))


# In[26]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))


# In[27]:


#knn algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(classification_report(y_test, pred_knn))


# In[29]:


conclusion = pd.DataFrame({'models': ["SVC","Random Forest","KNN"],
                           'accuracy': [accuracy_score(y_test,pred_svc),accuracy_score(y_test,pred_rfc),accuracy_score(y_test,pred_knn)]})
conclusion


# In[ ]:




