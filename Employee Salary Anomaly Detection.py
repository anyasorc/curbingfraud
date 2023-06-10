#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the dataset
df = pd.read_csv('D:/rawdata/Emp_salary.csv') # specify the path where your dataset is
df.head(5)


# In[3]:


# we will be selecting First_name, last_name, salary, department
df = df[['first_name', 'last_name', 'salary', 'department']]


# In[4]:


# check how many records are there in the dataset
df.shape


# In[5]:


# view all the records
df.head(15)


# In[6]:


# Select only salary for the isolation forest model test
df_salary = df[['salary']]
df_salary


# In[7]:


# Instantiate the model and fit the data to it
model=IsolationForest(n_estimators=1000, max_samples='auto', contamination=float(0.04),max_features=1.0, random_state=0)
model.fit(df_salary[['salary']])


# In[8]:


# Get the score and anomaly flag
df_salary['scores']=model.decision_function(df[['salary']])
df_salary['anomaly']=model.predict(df[['salary']])


# In[9]:


# view the data
df_salary


# In[10]:


# fetch all the anomalies
anomaly=df_salary.loc[df_salary['anomaly']==-1]
anomaly_index=list(anomaly.index)
anomaly.head(40)


# In[11]:


#merge the two dataframes to know which department's salary deviates from other departments
df_merged = pd.merge(df, df_salary, on=["salary"])
df_merged


# In[15]:


# Test your model
# 1 ==> The amount is within the expected salary range
# -1 ==> The amount deviates from the expected salary range
model.predict([[10000]])


# In[ ]:





# In[ ]:




