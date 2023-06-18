#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import json
import matplotlib.pyplot as plt
import plotly.express as px
import os
import ast
from sklearn.ensemble import IsolationForest
import pickle
warnings.filterwarnings('ignore')


# In[2]:


def writetopicke(nuban, data, full_path):
    modelpath = "D:\\curbingfraud\\CustTxnPatternModels\\"
    # the number of transaction days
    # nun_days_pattern = "30"
    # form the dataframe name variable to be used
    # txndays = "sum_"+nun_days_pattern+"days"
    model=IsolationForest(n_estimators=1000, max_samples='auto', contamination=float(0.04),max_features=1.0, random_state=0)
    #model = model.fit(data[[txndays]])
    model.fit(df[['transaction_amount','hour']])
    name_of_model = modelpath + nuban + "_txn_Pattern.pkl"
    #open a file, where you ant to store the data
    filename = open(name_of_model, 'wb')
    #save the model into a .pickel file
    pickle.dump(model, filename)
    # close the file
    scores_anomaly(data,model)
    get_anomaly(data)
    plot_anomaly_graph(data)
    filename.close()
    #removefile(full_path)


# In[3]:


def scores_anomaly(data,model):
    data['scores']=model.decision_function(data[['transaction_amount','hour']])
    data['anomaly']=model.predict(data[['transaction_amount','hour']])
    print(data)


# In[4]:


def get_anomaly(data):
    # fetch all the anomalies
    anomaly=data.loc[data['anomaly']==-1]
    anomaly_index=list(anomaly.index)
    #print(anomaly)


# In[5]:


def plot_anomaly_graph(data):
    # visualize the outcome for more clarity
    fig = px.scatter(data.reset_index(), x='hour', y='transaction_amount', color='anomaly', 
                 hover_data=['transaction_amount'], title='BANK TRANSACTION')
    fig.update_xaxes(
    rangeslider_visible=True,
    )
    fig.show()


# In[6]:


# read data from their respective folders
# create a list of our folder path   
cnt = 0
lst = ['data.1', 'data.2', 'data.3']  
# Calling DataFrame constructor on list  
dframe = pd.DataFrame(lst)  
# get the unique values from the list
df_unqiue_dir = dframe[0].unique()
# loop through each folder (data.1 etc) to train and generate customers unique model
for i in df_unqiue_dir:
    print(i)
    folder_to_view = "D:\\curbingfraud\\datasets\\"+i # i contains the folder names e.g data.1, data.2 ...
    dir_path = folder_to_view
    totalfile = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
    
    for file in os.listdir(folder_to_view):
        cnt+=1
        full_path = f'{folder_to_view}\{file}'
        filename, ext = os.path.splitext(file)
        if ext == '.csv':
            df = pd.read_csv(full_path, delimiter=',')
            print(f"Processing file number: {cnt} for transaction pattern in file {filename}") 
            writetopicke(filename, df, full_path)


# In[15]:


# test one of the model
acct = "0011168887"
modelpath = "D:\\curbingfraud\\CustTxnPatternModels\\"
name_of_model = modelpath + acct + "_txn_Pattern.pkl"
name_of_model

infile = open(name_of_model,'rb') #open the pickel file
model = pickle.load(infile, encoding='bytes') #read the pickel file
infile.close() #close the pickel file

prediction = model.predict([[500,5]])
prediction

#lets predict with the model to comfirm same


# In[ ]:




