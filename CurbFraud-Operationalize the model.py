#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# In[88]:


def getmodelpath(acct):
    # check if the account exist in the list of accounts
    # if it does then go and fetch the customer's model
    # else use generic model for the customer since it is assumed that the customer is new
    # and does not have a trained model yet.
    my_acct_list = [
    "0011138883",
    "0011148884",
    "0011168811",
    "0011168819",
    "0011168886",
    "0011168810",
    "0011168813",
    "0011118881",
    "0011168887",
    "0011128882",
    "0011168888",
    "0011168816",
    "0011168889",
    "0011168815",
    "0011168812",
    "0011168818",
    "0011168820",
    "0011158885",
    "0011168817",
    "0011168814"]
    
    search_item = acct
    
    if search_item in my_acct_list:
        val = 1
    else:
        val = -1
    
    if val > 0:
        #get the path to the model and assign to the variable modelpath
        modelpath = "D:\\curbingfraud\\CustTxnPatternModels\\"
        name_of_model = modelpath + acct + "_txn_Pattern.pkl"
        print(f"The name of the model gotten is {name_of_model} for account {acct}")
    else:
        # get the generic model path since the customer account does not have any model yet
        print(f"customer with acct {acct} does not have any transaction pattern and hence should be treated as a new customer")
        Generictxnpattern = "D:\\curbingfraud\\bankwidepatternmodel\\"
        name_of_model = Generictxnpattern + "General_txn_Pattern.pkl"
        print(f"The name of the model gotten is {name_of_model} for account {acct}")
    return name_of_model, val


# In[93]:


def processoutput(output,amt, acct):
    remark =""
    #check if the predicted outcome is -1
    if output == -1:
        remark = "The model predicts a TOTAL DEVIATION [Anomaly]  from customer's txn pattern for Account "         + acct + " with txn amount " + str(amt)
        rsp = {'amt': amt, 'acct': acct,
                  'predictedScore': str(output), 'remark': remark}
        return jsonify(predictedvalue = rsp)
    else:
        #prepare the json response
        remark = "The model predicts that customer txn pattern is OK for Account " + acct +         " with txn amount " + str(amt)
        rsp = {'amt': amt, 'acct': acct,
               'predictedScore': str(output), 'remark': remark}
        return jsonify(predictedvalue = rsp)


# In[90]:


app = Flask(__name__)
@app.route('/api/custxnpattern', methods=['POST'])
def predict():
    # read the data into its respective varibales
    data = request.get_json(force=True)
    acct = data['acct']
    amt = float(data['amt'])
    # check to ensure the account parameter is not empty
    if acct == '':
        remark = 'Kindly ensure that the account field is not empty'
        rsp = {'amt': amt, 'acct': acct,
               'predictedScore': '', 'remark': remark}
        #log into the log file
        return jsonify(predictedScore='', acct=acct, amt=amt, remark=remark)
    
    # check to ensure the amount parameter is not empty or less than or equal to 0
    if amt is None or amt <= 0:
        remark = 'Kindly ensure the amount field is not empty and that the value is >0'
        rsp = {'amt': amt, 'acct': acct,
                'predictedScore': '', 'remark': remark}
        #log into the log file
        return jsonify(predictedScore='', acct=acct, amt=amt, remark=remark)
    
    #get the name of the model.
    # note: the name_of_model variable can contain either the customer transaction pattern
    # or the generic model if the customer is new.
    name_of_model, val = getmodelpath(acct)
    print(f"This the model fetched ==> {name_of_model}")
    
    try:
        infile = open(name_of_model,'rb') #open the pickel file
        model = pickle.load(infile, encoding='bytes') #read the pickel file
        infile.close() #close the pickel file
    except:
        remark = "Unable to find file path for customer's " + acct + " transaction pattern"
        return jsonify(predictedScore='', acct=acct, amt=amt, remark=remark)
    
    #pass the amount and hour
    currentDateAndTime = datetime.now()
    txn_hour = currentDateAndTime.hour
    print(f"The transaction hour is ==> {txn_hour}")
    prediction = model.predict([[np.array(amt), np.array(txn_hour)]])
    
    # Take the first value of prediction
    output = prediction[0]
    print(f"The output gotten for account: {acct} with mdoel: {name_of_model} is {output}")
    return processoutput(output, amt, acct)


# In[91]:


if __name__ == '__main__':
    app.run(port=5009)


# In[ ]:




