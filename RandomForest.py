#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn import metrics


# In[6]:


from sklearn.tree import export_graphviz


# In[26]:


Hea=pd.read_csv("/home/student/Desktop/ml/random.csv")


# In[27]:


Hea.head()


# In[28]:


Hea.info()


# In[29]:


X=Hea[["Cloud3pm","Temp3pm"]]


# In[30]:


y=Hea[["Target"]]


# In[31]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)


# In[32]:


lm=RandomForestClassifier(n_estimators=100)


# In[33]:


lm.fit(X_train,y_train)


# In[34]:


prediction=lm.predict(X_test)


# In[35]:


print(metrics.accuracy_score(y_test, prediction))


# In[36]:


feature_cols=["Cloud3pm","Temp3pm"]


# In[38]:


estimator=lm.estimators_[4]


# In[39]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


# In[40]:


export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_cols,
                class_names = ["0","1","2","12"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)


# In[41]:


# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


# In[42]:


# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# In[ ]:




