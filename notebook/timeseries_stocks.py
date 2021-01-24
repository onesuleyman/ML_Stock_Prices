"""
#why TIME SERIES MODEL?
#Because we want to model stock prices correctly, 
# so as a stock buyer you can reasonably decide when to buy stocks and when to sell them to make profit. 
#  Therefore,
#   You need a good machine learning model that can look at hte history of a sequence of data 
#       and correctly predict what the future elements of the sequence are going to be.
"""

#library
from pandas_datareader import data 
import matplotlib.pyplot as plt
import pandas as pd 
from datetime import datetime as dt  
import urllib.request, json 
import os 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
#

df = pd.read_csv("/Users/suleymanekiz/Desktop/personal_projects/ML_Stock_Prices/data/archive/Stocks/hpq.us.txt", delimiter=',', usecols=[ 'Date', 'Open', 'High', 'Low', 'Close'])
print('Loaded data from the Kaggle repository')

#### DATA EXPLORATION

#sort dataframe by date
#   this is crucial for time series model
#df = df.sort_values('Date')
#print(df.head())
#print(df.info)
#       DATE SORTED!

#### DATA VISUALIZATION

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500], rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
#plt.show()#

#### SPLIT DATA
#  into training-test set
#take the average of high and low prices on a day and store as mid_prices

high_prices = df.loc[:, 'High'].values
low_prices = df.loc[:, 'Low'].values
mid_prices = (high_prices+low_prices)/2.0

#split train/test
train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

#### NORMALIZE DATA
# Scale data between 0 and 1
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

#train the scaler with training data and smooth data
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size:,:]
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

#normalize the last
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

#reshape both train and test data
train_data = train_data.reshape(-1)

#normalize test data
test_data = scaler.transform(test_data).reshape(-1)

#### AVG. SMOOTHING
#perform average smoothing
#to have less raggedness of data in stock prices
#produces a smoother curve
EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

#visual and test purpose
all_mid_data = np.concatenate([train_data, test_data], axis=0)

