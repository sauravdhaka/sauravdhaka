#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[4]:


credits.head()


# In[5]:


movies.head()


# In[6]:


credits['id'] = credits['movie_id']
credits.drop('movie_id',axis=1,inplace=True)


# In[7]:


df = movies.merge(credits,on='id')
df.drop(['homepage', 'title_x', 'title_y', 'status','production_countries'],axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


df['overview'][0]


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


# In[11]:


tfidf = TfidfVectorizer(stop_words='english',analyzer='word',min_df=3,strip_accents='unicode'
                         ,ngram_range=(1,3),token_pattern=r'\w{1,}',max_features=None)

df['overview'].fillna('',inplace=True)
vector = tfidf.fit_transform(df['overview'])
vector


# In[12]:


vector.shape


# In[13]:


sigmoid = sigmoid_kernel(vector,vector)

sigmoid


# In[14]:


sigmoid[0]


# In[15]:


sigmoid.shape


# In[17]:


index = pd.Series(df.index,index=df['original_title']).drop_duplicates()
index


# In[18]:


df.head()


# In[19]:


index['Avatar']


# In[20]:


list(enumerate(sigmoid[index['Avatar']]))


# In[21]:


def content(title,sigmoid=sigmoid):
        position = index[title]
        score = sorted(list(enumerate(sigmoid[position])),key=lambda x:x[1],reverse=True)
        indices = score[1:11]
        movie_indices = [i[0] for i in indices]
        # Top 10 most similar movies
        return df['original_title'].iloc[movie_indices]


# In[22]:


content('Avatar')


# In[23]:


content('Spectre')


# In[24]:


content('The Dark Knight Rises')


# In[25]:


df['original_title'][0]


# In[ ]:




