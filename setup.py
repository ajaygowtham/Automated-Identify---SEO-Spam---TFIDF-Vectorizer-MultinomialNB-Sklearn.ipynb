
# coding: utf-8

# In[327]:


import numpy as np
import pandas as pd
import matplotlib
from pylab import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score


# In[328]:


df=pd.read_csv('data1/spamdata.csv',names=['Status','Message'])


# In[329]:


df.head()


# In[330]:


len(df[df.Status=='Bad'])


# In[331]:


df.loc[df["Status"]=='Good', "Status"]=1
df.loc[df["Status"]=='Bad', "Status"]=0


# In[332]:


df.head()


# In[333]:


df_x=df["Message"]
df_y=df["Status"]


# In[334]:


df_x


# In[335]:


cv1=TfidfVectorizer(min_df=1, stop_words='english')


# In[336]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# In[337]:


x_traincv = cv.transform(["Download hack tools", "How to hack Facebook", "Watch Pirated movies online"])


# In[338]:


x_traincv.toarray()


# In[339]:


cv.get_feature_names()


# In[340]:


cv1=CountVectorizer()


# In[341]:


x_traincv=cv1.fit_transform((x_train).values.astype('str'))


# In[342]:


a=x_traincv.toarray()


# In[343]:


a


# In[344]:


a[0]


# In[345]:


cv1.inverse_transform(a[0])


# In[346]:


x_train.iloc[0]


# In[347]:


mnb=MultinomialNB()


# In[348]:


y_train=y_train.astype('int')


# In[349]:


mnb.fit(x_traincv, y_train)


# In[350]:


x_testcv=cv1.transform(x_test)


# In[351]:


pred=mnb.predict(x_traincv)


# In[352]:


pred


# In[353]:


actual=np.array(y_test)


# In[354]:


actual


# In[355]:


count=1


# In[356]:


for i in range (len(pred)):
    if pred[i]==actual[i]:
        count=count+ 1


# In[357]:


count


# In[358]:


len(pred)


# In[359]:


x_test.iloc[0]

