#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import the required library
import warnings
import pandas as pd
from datetime import timedelta


# In[15]:


# Load the bank transaction dataset
# specify the path where your dataset is
df = pd.read_csv('D:/rawdata/tbl_bank_transactions.csv',dtype={'account_number':str,'transaction_amount':float}) 
df.head(3)


# In[16]:


# confirm how many records exist and the number of rows and columns
df.shape


# In[17]:


# we will be working with account_number, transaction_date_time, and transaction_amount
df = df[['account_number','transaction_date_time', 'transaction_amount', 'transaction_type']]
#peek into the data
df.head(5)


# In[18]:


#lets convert the transaction_date_time from object to a proper date time
df['transaction_date_time']=pd.to_datetime(df['transaction_date_time'])
df


# In[19]:


# confirm the data types
df.dtypes


# In[20]:


# Let us define a pattern to use for our fraud detection. 
# create a new column called hour and extract hourly value from the transaction_date_time
df['hour']=df.transaction_date_time.dt.hour
# create a new column called weekday and extract the weekday value from transaction_date_time
df['weekday']=pd.Categorical(df.transaction_date_time.dt.strftime('%A'), categories=['Monday','Tuesday','Wednesday',
                                                                                     'Thursday','Friday','Saturday', 
                                                                                     'Sunday'], ordered=True)
df


# In[22]:


df.dtypes


# In[23]:


# check for NAN in the amount 
print(df.isna().sum())


# In[25]:


# function to generate each customers transaction pattern by account
# and write into folders
def gen_txn_pattern_per_acct(df):
    # get the unique transaction accounts
    # this will end up getting 20 unique accounts
    # since our dataset contains just 20 customers account with transactions
    df_unqiue_accts = df.account_number.unique()
    # get the path to write data.  This is the path where the folders data.1, data.2, data.3 were created.
    writedata = "D:\\curbingfraud\\datasets\\"
    # set the txn pattern number of days for pattern generation per account
    # as approved by management
    #num_of_days_txn_pattern = "30"
    # form the dataframe name variable to be used
    # txndays = "sum_"+num_of_days_txn_pattern+"days" # this will become sum_30days
    cnt = 0
    datafolder = ''
    total = 0
    for i in df_unqiue_accts:
        total += 1
        cnt += 1
        # pick the account to start generating txn pattern where transaction_type = withdrawal
        df_per_acct = df.query('account_number == @i & transaction_type =="withdrawal"')
        # sort the value by input date and set the index to input date
        # the set_index() method allows one or more column values become the row index.
        # Note: I used the \ to move the rest of the code to a new line.
        df_txn_pattern = df_per_acct.sort_values(by=['account_number', 'transaction_date_time']).        set_index('transaction_date_time')
        
        # get the sum of the previous n days transaction amount based on customers account
        # Note: I used the \ to move the rest of the code to a new line.
        # df_txn_pattern[txndays] = df_txn_pattern.groupby('account_number')['transaction_amount'].\
        # transform(lambda s: s.rolling(timedelta(days=int(num_of_days_txn_pattern))).sum())
        # assign the value into new dataframe named final_df
        final_df = df_txn_pattern
        # check if the cnt is 3 then reset cnt 1
        # cnt ==3 means it has gotten to the last folder
        # and there is a need to start writing into folder 1 and proceed to the next folder
        # until we are done splitting.
        if cnt == 1:
            datafolder = "data.1"
            print(f"Writing data for {i} into folder path {datafolder}. item {str(total)} of ==> {str(len(df_unqiue_accts))}")
        elif cnt == 2:
            datafolder = "data.2"
            print(f"Writing data for {i} into folder path {datafolder}. item {str(total)} of ==> {str(len(df_unqiue_accts))}")
        elif cnt == 3:
            datafolder = "data.3"
            print(f"Writing data for {i} into folder path {datafolder}. item {str(total)} of ==> {str(len(df_unqiue_accts))}")
            cnt = 0
        final_df.to_csv(writedata + datafolder + "\\" + i + '.csv', index = False)

    return total 
#*******************************************************


# In[26]:


# generate txn pattern file per account
val = gen_txn_pattern_per_acct(df)
print(f"Data processing completed... with total item {str(val)}")


# In[ ]:




