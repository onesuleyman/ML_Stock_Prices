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



########### ONE DAY AHEAD PREDICTION with AVERAGING

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, 'Y-%m-%d').date() + dt.timedelta(days=1)
    else: 
        date= df.loc[pred_idx, 'Date']
    
    std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx])**2)
    std_avg_x.append(date)
print('MSE error for standard averaging: %.5f'%(0.5 * np.mean(mse_errors))) # 0.00418

## plot ONE DAY AHEAD PREDICTION with AVERAGING

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data,color='b', label='True')
plt.plot(range(window_size,N),std_avg_predictions, color= 'orange', label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show() #the plot shows that the model is doing good for very short predictions (one day ahead) This behaviour is sensible since the stock does not change overnight. 


###### Exponential Moving Average
window_size = 100
N = train_data.size
run_avg_predictions = []
run_avg_x = []
mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors))) #0.00003

##plot Exponential Moving Average
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data,color='b', label='True')
plt.plot(range(0,N),run_avg_predictions, color= 'orange', label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show() #line fits perfectly (follows the label TRUE) justified by the very low MSE



####### LSTM (Long Short-Term Memory model)

class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)
            
            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length
        
        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_incides(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments, self._prices_length-1))

dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui, (dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ', dat)
    print('\n\tOutput: ', lbl)



##### Defining Hyperparameters

D = 1 #dimensionality of data
num_unrollings = 50 #number of time steps you look into the future
batch_size = 500 #number of samples in batch 
num_nodes = [200,200,150] #number of hidden nodes(neurons) in each layer of deep LSTM stack
n_layers = len(num_nodes) #number of layers
dropout = 0.2 #dropout amount

#tf.reset_default_graph() #import when running multiple times

##### Defining INPUTS & OUTPUTS

#input data
train_inputs, train_outputs = [], []

#unroll the input over time defining placeholders for each step
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D], name='train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))



##### Defining Parameters of the LSTM and Regression Layer
lstm_cells = [
    tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
    state_is_tuple = True, 
    initializer= tf.contrib.layers.xavier_initializer())
for li in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
    lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
) for lstm in lstm_cells]
drop_multi_cll = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contriv.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w', shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', initializer = tf.random_uniform([1], -0.1,0.1))



############ CALCULATE LSTM OUTPUT AND FED TO REGRESSION LAYER FOR FINAL PREDICTION ###############

#create cell state and hidden state variables to maintain the state of the LSTM
c, h = [], []
initial_state = []
for li in range(n_layers):
    c.append(tf.Vriable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Vriable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

#tensor transfformations due to specific format dynamic_rnn requires in its output
all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs], axis=0)

#all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,w,b)

split_outputs = tf.split(all_outputs,num_unrollings,axis=0)



###### LOSS CALCULATION and OPTIMIZER

