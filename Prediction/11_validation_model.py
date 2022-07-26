#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:53:28 2021

@author: chay
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Layer
import keras.backend as K
from keras import callbacks
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)

#num_prediction = 5
split_percent = 0.90
look_back = 3
num_epochs =150
TITLE = 'energy transfer'

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
  
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

df = pd.read_excel('keyword_frequency_half_year_weight.xlsx', sheet_name='freq', header =0) 
raw_seq = df[TITLE].values

# without attetion 
seq_data = raw_seq
seq_data = seq_data.reshape((-1,1))

split = int(split_percent*len(seq_data))
seq_train = seq_data[:split]
seq_test = seq_data[split:]

train_generator = TimeseriesGenerator(seq_train, seq_train, length=look_back, batch_size=8)     
test_generator = TimeseriesGenerator(seq_test, seq_test, length=look_back, batch_size=2)

model = Sequential()
model.add(LSTM(300, activation='relu', input_shape=(look_back,1)))
model.add(RepeatVector(1))
model.add(LSTM(300, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#model.summary()
hist_l=model.fit(train_generator, epochs=num_epochs, verbose=0, shuffle=False)
prediction = model.predict(test_generator)

loss_value_l = hist_l.history["loss"]
plt.plot(loss_value_l,  label='train_lstm')


seq_train= seq_train.reshape((-1))
seq_test = seq_test.reshape((-1))
prediction = prediction.reshape((-1))

mse = mean_squared_error(seq_test[look_back:], prediction)
rmse = sqrt(mse)
print('RMSE: %f' % rmse)
print('MSE: %f' % mse)
i =0
sum_differ =0
mean_differ =0
for pred in prediction:
    if(seq_test[(look_back)+i])!=0:
        differ = abs(pred - seq_test[(look_back)+i])/seq_test[(look_back)+i]
    else:
        differ = 0
    sum_differ += differ
    mean_differ = sum_differ/(len(prediction))
    valid_value =mean_differ    
    i += 1
print('MAPE: %f' % valid_value)
seq_test_plt = np.append(seq_train[-1:], seq_test)
prediction_plt = np.append(seq_test_plt[:(look_back+1)], prediction)

# with attetion 

seq_data = raw_seq
seq_data = seq_data.reshape((-1,1))

split = int(split_percent*len(seq_data))
seq_train = seq_data[:split]
seq_test = seq_data[split:]

train_generator = TimeseriesGenerator(seq_train, seq_train, length=look_back, batch_size=8)     
test_generator = TimeseriesGenerator(seq_test, seq_test, length=look_back, batch_size=2)

model_att = Sequential()
model_att.add(LSTM(300, activation='relu', input_shape=(look_back,1),return_sequences=True))
model_att.add(attention()) 
model_att.add(RepeatVector(1))
model_att.add(LSTM(300, activation='relu', return_sequences=True))
model_att.add(TimeDistributed(Dense(1)))
model_att.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#model_att.summary()
hist = model_att.fit(train_generator, epochs=num_epochs, verbose=0, shuffle=False)
prediction_att = model_att.predict(test_generator)

loss_value = hist.history["loss"]
plt.plot(loss_value, label='train_lstm+attetion')
plt.title(TITLE)
plt.legend(loc = 'upper left')
plt.show()

seq_train= seq_train.reshape((-1))
seq_test = seq_test.reshape((-1))
prediction_att = prediction_att.reshape((-1))

mse_att = mean_squared_error(seq_test[look_back:], prediction_att)
rmse_att = sqrt(mse_att)
print('RMSE_ATT: %f' % rmse_att)
print('MSE_ATT: %f' % mse_att)

i_att =0
sum_differ_att =0
mean_differ_att =0
for pred_att in prediction_att:
    if(seq_test[(look_back)+i_att])!=0:
        differ_att = abs(pred_att - seq_test[(look_back)+i_att])/seq_test[(look_back)+i_att]
    else:
        differ_att = 0
    sum_differ_att += differ_att
    mean_differ_att = sum_differ_att/(len(prediction_att))
    valid_value_att =mean_differ_att    
    i_att += 1
print('MAPE_ATT: %f' % valid_value_att)

seq_test_att_plt = np.append(seq_train[-1:], seq_test)
prediction_att_plt = np.append(seq_test_att_plt[:(look_back+1)], prediction_att)

plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize']= 14
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.titlesize']= 20
plt.rcParams['axes.labelsize']= 14
plt.title(TITLE)
plt.xlabel("Published Year")
plt.ylabel("Frequency Scale")
plt.plot(range(len(seq_train)), seq_train,'.-', color ='black',label = 'Train Data')
plt.plot(range(len(seq_train)-1, len(seq_train)+len(seq_test)), seq_test_plt, '--', color ='black',label = 'Ground Truth')
plt.plot(range(len(seq_train)-1, len(seq_train)+len(seq_test)), prediction_plt,'.-',color ='red', label = 'LSTM Prediction')
plt.plot(range(len(seq_train)-1, len(seq_train)+len(seq_test)), prediction_att_plt,'.-',color ='blue', label = 'LSTM+Attetion Prediction')
plt.xticks([0,10,20,30,40,50,60,70,80], ['1981', '1986','1991', '1996', '2001', '2006', '2011','2016', '2021']) #year
plt.legend(loc = 'upper left')





















