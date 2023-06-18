#!/usr/bin/env python
# coding: utf-8

# In[22]:


#import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.express as px
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[23]:


# Load the bank transaction dataset
# specify the path where your dataset is
df = pd.read_csv('D:/rawdata/tbl_bank_transactions.csv',dtype={'account_number':str,'transaction_amount':float}) 
df.head(5)


# In[24]:


# confirm how many records exist and the number of rows and columns
df.shape


# In[25]:


# we will be working with account_number, transaction_date_time, and transaction_amount
df = df[['account_number','transaction_date_time', 'transaction_amount']]
#peak into the data
df.head(10)


# In[26]:


# confirm the data types
df.dtypes


# In[27]:


#lets convert the transaction_date_time from object to a proper date time
df['transaction_date_time']=pd.to_datetime(df['transaction_date_time'])


# In[28]:


df.dtypes


# In[29]:


# We will resample the time-series dataset and aggregate it to hourly intervals
df=df.set_index('transaction_date_time').resample("H").mean().reset_index()
pd.options.display.float_format = '{:.2f}'.format
df


# In[30]:


#check for NAN in the amount 
df.isna().sum()


# In[31]:


# drop the NaN
df = df.dropna()
df = df.dropna(axis=0)


# In[32]:


#check for NAN in the amount 
df.isna().sum()


# In[33]:


# Let us define a pattern to use for our fraud detection. 
# create a new column called hour and extract hourly value from the transaction_date_time
df['hour']=df.transaction_date_time.dt.hour
# create a new column called weekday and extract the weekday value from transaction_date_time
df['weekday']=pd.Categorical(df.transaction_date_time.dt.strftime('%A'), categories=['Monday','Tuesday','Wednesday',
                                                                                     'Thursday','Friday','Saturday', 
                                                                                     'Sunday'], ordered=True)
df


# In[34]:


#Plotting line charts
df[['transaction_amount','weekday']].groupby('weekday').mean().plot()


# In[35]:


df_Thursday = df.query("weekday == 'Thursday'")
df_Friday = df.query("weekday == 'Friday'")
df_Saturday = df.query("weekday == 'Saturday'")
df_Sunday = df.query("weekday == 'Sunday'")


# In[36]:


#plot transaction_amount vs hour
df[['transaction_amount','hour']].groupby('hour').mean().plot()


# In[37]:


# Lets define a transaction pattern with the transaction amount and hour
# and feed the data to the Isolation Forest model
# Instantiate the model and fit the data into the Isolation Forest Model
# with a contamination value of 0.04 (4%) for the contamination parameter
model=IsolationForest(n_estimators=1000, max_samples='auto', contamination=float(0.04),max_features=1.0, random_state=0)
model.fit(df[['transaction_amount','hour']])


# In[38]:


df['scores']=model.decision_function(df[['transaction_amount','hour']])
df['anomaly']=model.predict(df[['transaction_amount','hour']])
df


# In[39]:


# fetch all the anomalies
anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
anomaly


# In[40]:


# visualize the outcome for more clarity
fig = px.scatter(df.reset_index(), x='hour', y='transaction_amount', color='anomaly', 
                 hover_data=['transaction_amount'], title='BANK TRANSACTION')
fig.update_xaxes(
    rangeslider_visible=True,
)
fig.show()


# In[41]:


#lets predict with the model to comfirm same
model.predict([[26000.00,0]])


# In[43]:


#Generate generic model for the bank
modelpath = "D:\\curbingfraud\\bankwidepatternmodel\\"
name_of_model = modelpath + "General_txn_Pattern.pkl"
#open a file, where you ant to store the data
filename = open(name_of_model, 'wb')
#save the model into a .pickel file
pickle.dump(model, filename)
# close the file
filename.close()


# In[ ]:




