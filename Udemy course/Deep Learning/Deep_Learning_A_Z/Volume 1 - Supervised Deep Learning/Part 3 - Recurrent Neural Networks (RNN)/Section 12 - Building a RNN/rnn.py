
#import libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#import training set
data_train=pd.read_csv('Google_Stock_Price_train.csv')
training_set=data_train.iloc[:,1:2].values
sc=MinMaxScaler(feature_range=(0,1))

training_set_scaled=sc.fit_transform(training_set)
X_train=[]
y_train=[]

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train,y_train=np.array(X_train),np.array(y_train)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))




























