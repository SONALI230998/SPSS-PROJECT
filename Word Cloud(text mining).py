#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install wordcloud')


# In[3]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 


# In[4]:


import numpy as np


# In[5]:


from os import path
from PIL import Image


# In[6]:


df = pd.read_csv("E:\\data science project\\Wine Reviews Data\\winemag-data_first150k.csv", index_col=0)


# In[7]:


df.head()


# In[9]:


print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))

#print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
#                                                                           ", ".join(df.variety.unique()[0:5])))

#print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
     


# In[10]:


print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                   ", ".join(df.variety.unique()[0:5])))


# In[11]:


print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
                                                                                      ", ".join(df.country.unique()[0:5])))


# In[12]:


df[["country", "description","points"]].head()


# In[13]:


country = df.groupby("country")


# In[14]:


country.describe().head()


# In[15]:


country.mean().sort_values(by="points",ascending=False).head()


# In[16]:


plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()


# In[17]:


plt.figure(figsize=(15,10))
country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()


# In[18]:


text = df.description[0]


# In[19]:


wordcloud = WordCloud().generate(text)


# In[20]:


# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[21]:


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[23]:


# Save the image in the img folder:
wordcloud.to_file("E:\data science project\Wine Reviews Data/first_review.png")


# In[24]:


text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[25]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




