
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib qt
from scipy import signal

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


"""
bin1 = dill.load(open("all_bins/r_bin_12227.dill", "rb"))

all_bins = []

for i in range(1, 153584):
    bin_name = "all_bins/r_bin_" + str(i) + ".dill"
    curr_bin = dill.load(open(bin_name, "rb"))
    if(curr_bin[3] == -1):
        bin_data = curr_bin[2]
        if(bin_data.size == 4095):
            bin_data = np.append(bin_data, 0)
        if(bin_data.size == 8192):
            bin_data = bin_data[:4096]
        bin_data = np.append(bin_data, curr_bin[1])
        all_bins.append(bin_data)

newData = pd.DataFrame(all_bins)

newData.to_csv("all_quakes.csv")
"""

data = pd.read_csv("all_quakes.csv")
X = data.iloc[:, 1:4097].values
y = data.iloc[:, 4097].values
X = X.astype(float)

X_fft2 = np.zeros((153567,4096), dtype = np.complex64)
X_fft2 = np.fft.fft2(X, (153567, 4096))



X_fft = np.zeros((153567,4096), dtype = np.complex64)
fftArr = np.array([])
for i in range(0, 153567):
    X_fft[i] = np.fft.fft(X[i])
    #fftArr = np.append(fftArr, X_fft[i])

X_lot = np.array([])
for i in range(0, 100):
    X_lot = np.append(X_lot, X[i])

f, t, Sxx = signal.spectrogram(x = X_fft2[2])
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


a = X[0]
b = X_fft2[2]

plt.plot(b)
plt.plot(X_fft[0])
plt.plot(X_fft[1156])

























# all 17 starts to quakes
quake_starts = [0, 1380, 12225, 25551, 33873, 45802, 53371, 60004, 75141, 82570, 91626, 102364, 112724, 121020, 129069, 142932, 151821]


feature_data = pd.read_csv("features_including_fft.csv")
X_features = feature_data.iloc[:, 1:].values

sc_X = StandardScaler()
X_features = sc_X.fit_transform(X_features)


# 5.67 avg with 3.04 mean error
"""
# find all the bins that are the start of a quake
for i in range(1, 153584):
    bin_name = "all_bins/r_bin_" + str(i) + ".dill"
    curr_bin = dill.load(open(bin_name, "rb"))
    if(curr_bin[3] != -1):
        print(i)
"""

#not 153584?
# get indexes of training data so no 36 bins goes over a quake
indexs = np.array([])
for start in range(0, 35):
    for i in range(start, 129068, 36):
        need_break = False
        for j in range(1, 16):
            if(i < quake_starts[j] and i + 35 > quake_starts[j]):
                need_break = True
        if(need_break):
            continue
        indexs = np.append(indexs, i)


for i in range(0, 129068):
    need_break = False
    for j in range(1, 16):
        if(i < quake_starts[j] and i + 35 > quake_starts[j]):
            need_break = True
    if(need_break):
        continue
    indexs = np.append(indexs, i)



indexs = indexs.astype(np.int32)
# shuffles indexes
np.random.shuffle(indexs)
# adds in missing values between indexes
y_data = np.zeros((indexs.size, 1))
all_indexs = np.zeros((indexs.size, 36))
for i in range(0, indexs.size):
    y_data[i] = y[indexs[i]+35]
    all_indexs[i] = np.arange(indexs[i], indexs[i]+36)

all_indexs.resize((indexs.size, 36))
all_indexs = all_indexs.astype(np.int32)
    

data_in3d = np.zeros((indexs.size, 36, 31))

for i in range(0, indexs.size):
    for j in range(0, 36):
        data_in3d[i,j,:] = X_features[all_indexs[i,j]]




model = Sequential()

model.add(TimeDistributed(Dense(units = 256, activation = 'relu', kernel_initializer = 'uniform'), input_shape = (36, 12)))
model.add(TimeDistributed(Dense(units = 128, activation = 'relu', kernel_initializer = 'uniform')))
model.add(TimeDistributed(Dense(units = 128, activation = 'relu', kernel_initializer = 'uniform')))
model.add(TimeDistributed(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform')))
model.add(LSTM(units = 8, input_shape = (36, 64), kernel_initializer = 'uniform'))

model.add(Dense(units = 1, kernel_initializer = 'uniform'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

model = load_model("actually_working_Poisson.h5")


model.fit(data_in3d, y_data, batch_size = 10000, epochs = 10)

y_pred = model.predict(data_in3d)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("y_pred_df.csv")
y_test_df = pd.DataFrame(y_data)
y_test_df.to_csv("y_test_df.csv")

model.save("actually_working_Poisson_George_is_the_best.h5")
model = load_model("featureRNN_v1.h5")

X = np.absolute(X)

X_features = np.zeros((153567, 12))
for i in range(0, 153567):
    X_features[i, 0] = np.mean(X[i])
    X_features[i, 1] = np.median(X[i])
    X_features[i, 2] = np.std(X[i])
    X_features[i, 3] = np.max(X[i])
    X_features[i, 4] = np.min(X[i])
    X_features[i, 5] = np.var(X[i])
    X_features[i, 6] = np.ptp(X[i]) #Peak-to-peak is like range
    X_features[i, 7] = np.percentile(X[i],q=10) 
    X_features[i, 8] = np.percentile(X[i],q=25) #We can also grab percentiles
    X_features[i, 9] = np.percentile(X[i],q=50)
    X_features[i, 10] = np.percentile(X[i],q=75)
    X_features[i, 11] = np.percentile(X[i],q=90)


feature_data = pd.DataFrame(X_features)
feature_data.to_csv("data_abs_features.csv")








