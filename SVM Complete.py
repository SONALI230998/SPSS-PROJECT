#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split


# In[65]:


from sklearn import svm


# In[66]:


import matplotlib.pyplot as plt


# In[67]:


from sklearn import metrics


# In[68]:


Adv=pd.read_csv("C:\\Users\\cadd\\Desktop\\heart.csv")


# In[69]:


Adv.head()


# In[70]:


Adv.info()


# In[71]:


X=Adv[["trestbps","chol"]]


# In[86]:


Adv.describe()


# In[72]:


y=Adv[["target"]]


# In[73]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)


# In[75]:


lm=svm.SVC(kernel='linear')


# In[76]:


lm.fit(X_train, y_train)


# In[77]:


prediction=lm.predict(X_test)


# In[78]:


prediction=np.where(prediction>0.5,1,0)


# In[79]:


print(metrics.accuracy_score(y_test, prediction))


# In[80]:


lm.support_vectors_


# In[150]:


X1=Adv['trestbps']
X2=Adv['chol']
X_training=np.array(list(zip(X1,X2)))
X_training


# In[152]:


y_training=Adv['target']
y_training


# In[153]:


target_names=['0','+1']
target_names


# In[154]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)
plt.xlabel('X1')
plt.ylabel('X2');
plt.savefig('chart0.png')


# In[155]:


svc = svm.SVC(kernel='linear').fit(X_training,y_training)
svc


# In[156]:


svc.get_params(True)


# In[157]:


lbX1=math.floor(min(X_training[:,0]))-1
ubX1=math.ceil(max(X_training[:,0]))+1
lbX2=math.floor(min(X_training[:,1]))-1
ubX2=math.ceil(max(X_training[:,1]))+1
[lbX1,ubX1,lbX2,ubX2]


# In[168]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)

X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=['k'], linestyles=['-'],levels=[0])

plt.title('Linearly Separable')
plt.savefig('chart1.png')


# In[170]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='g',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)
X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=['k','k','k'], linestyles=['--','-','--'],levels=[-1,0,1])
plt.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],s=120,facecolors='none')
plt.scatter(X_training[:,0],X_training[:,1],c=y_training,s=50,alpha=0.95);

plt.title('Margin and Support Vectors')
plt.savefig('chart2.png')


# In[160]:


svc.n_support_


# In[161]:


svc.support_


# In[ ]:




