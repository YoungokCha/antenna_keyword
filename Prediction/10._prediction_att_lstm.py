#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:57:14 2021

@author: chay
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Layer
import keras.backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)

num_prediction = 10
split_percent = 0.90
look_back = 3
num_epochs =150
moving_no = 3

def predict(num_prediction, model):
    prediction_list = seq_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
        
    prediction_list = prediction_list[look_back:]
        
    return prediction_list

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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
raw_seq = df['energy transfer'].values

# ##attention model
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
model_att.summary()

model_att.fit(train_generator, epochs=num_epochs, verbose=1, shuffle=False)
prediction = model_att.predict(test_generator)

seq_train= seq_train.reshape((-1))
seq_test = seq_test.reshape((-1))
prediction = prediction.reshape((-1))
seq_data = seq_data.reshape((-1))
att_forecast = predict(num_prediction, model_att)

#### without lstm model

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
model.summary()

model.fit(train_generator, epochs=num_epochs, verbose=1, shuffle=False)
prediction = model.predict(test_generator)

seq_train= seq_train.reshape((-1))
seq_test = seq_test.reshape((-1))
prediction = prediction.reshape((-1))
seq_data = seq_data.reshape((-1))
lstm_forecast = predict(num_prediction, model)


all_lstm_seq_data = np.append(seq_data, lstm_forecast)
lstm_start_data = seq_data[0:moving_no-1]
lstm_moving_data = np.append(lstm_start_data, moving_average(all_lstm_seq_data, moving_no))

lstm_past = lstm_moving_data[0:len(seq_data)]
print(len(lstm_past))
lstm_future = lstm_moving_data[len(seq_data)-1:]
print(len(lstm_future))


all_att_seq_data = np.append(seq_data, att_forecast)
att_start_data = seq_data[0:moving_no-1]
att_moving_data = np.append(att_start_data, moving_average(all_att_seq_data, moving_no))

att_past = att_moving_data[0:len(seq_data)]
print(len(att_past))
att_future = att_moving_data[len(seq_data)-1:]
print(len(att_future))

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

#plt.figure(figsize =(10,6))
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

#sns.set_style("ticks")
plt.title("'energy transfer'")
plt.xlabel("Published Year")
plt.ylabel("Frequency Scale")
plt.plot(range(len(lstm_past)), lstm_past,'-', color ='black',label = 'Past Data')
plt.plot(range(len(lstm_past)-1, len(lstm_past)+len(lstm_future)-1), lstm_future , '-', color ='red', label = 'LSTM Prediction') #MA
plt.plot(range(len(lstm_past)-1, len(lstm_past)+len(att_future)-1), att_future , '-', color ='blue', label = 'LSTM+Attention Prediction') #MA
#plt.xticks([0,8,16,24,32,40,48,56,64,72,80,88,96], ['2002', '2004','2006', '2008', '2010', '2012', '2014','2016', '2018','2020','2022','2024', '2026']) #quarter
#plt.xticks([0,5,10,15,20,25,30,35,40,45], ['1981', '1986','1991', '1996', '2001', '2006', '2011','2016', '2021','2026']) #year
plt.xticks([0,10,20,30,40,50,60,70,80,90], ['1981', '1986','1991', '1996', '2001', '2006', '2011','2016', '2021','2026']) #half year
plt.legend(loc = 'upper left')
#plt.show()








