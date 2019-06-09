

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error
import dill

bin1 = dill.load(open("r_bin_1.dill", "rb"))

all_bins = []
#50085878
for i in range(1, 30000):
    bin_name = "r_bin_" + str(i) + ".dill"
    curr_bin = dill.load(open(bin_name, "rb"))
    bin_data = curr_bin[2]
    if(bin_data.size == 4095):
        bin_data = np.append(bin_data, 0)
    bin_data = np.append(bin_data, curr_bin[1])
    all_bins.append(bin_data)

sqArr = np.array(all_bins)
newData = pd.DataFrame(all_bins)

newData.to_csv("first_2+_quakes.csv")

data = pd.read_csv("first_2+_quakes.csv")
X = newData.iloc[:, 1:4097].values
y = newData.iloc[:, 4097].values

X = X.astype(np.int16)
y = y.astype(np.float64)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Sequential()

model.add(Dense(units = 8192, activation = 'relu', kernel_initializer = 'uniform', input_dim = 4096))
model.add(Dropout(.2))
model.add(Dense(units = 4096, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dropout(.2))
model.add(Dense(units = 4096, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dropout(.2))
model.add(Dense(units = 4096, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dropout(.2))
model.add(Dense(units = 4096, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dropout(.2))
model.add(Dense(units = 2048, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dense(units = 1024, activation = 'relu', kernel_initializer = 'uniform'))

model.add(Dense(units = 1, kernel_initializer = 'uniform'))

model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 1000, epochs = 100, verbose = 2)

y_pred = model.predict(X_test)

mean_absolute_error(y_test, y_pred)