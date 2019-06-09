
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib qt
import scipy
import dill

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, LSTM, RNN, TimeDistributed
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras import optimizers, Model


data = pd.read_csv("train_chunk1.csv")
"""
data = pd.read_csv("all_quakes.csv")
X = data.iloc[:, 1:4097].values
y = data.iloc[:, 4097].values
X = X.astype(np.int16)

X = np.absolute(X)

X_features = np.zeros((153567, 31))

for i in range(0, 153567):
    X_features[i, 0] = np.mean(X[i])
    X_features[i, 1] = np.median(X[i])
    X_features[i, 2] = np.std(X[i])
    X_features[i, 3] = np.max(X[i])
    X_features[i, 4] = np.var(X[i])
    X_features[i, 5] = np.ptp(X[i])
    X_features[i, 6] = np.percentile(X[i], q = 10)
    X_features[i, 7] = np.percentile(X[i], q = 25)
    X_features[i, 8] = np.percentile(X[i], q = 50)
    X_features[i, 9] = np.percentile(X[i], q = 75)
    X_features[i, 10] = np.percentile(X[i], q = 90)
    X_features[i, 11] = scipy.stats.entropy(X[i])
    X_features[i, 12] = scipy.stats.kurtosis(X[i])
    X_features[i, 13] = scipy.stats.skew(X[i])
    
    if (i <= 153566):
        X_features[i, 14] = np.correlate(X[i], X[i + 1]) #Corr of two consecutive bins
        
    if (i <= 153556):
        X_features[i, 15] = np.correlate(X[i], X[i + 10]) #Corr of 10 consecutive bins

X_fft = np.zeros((153567,4096))
fftArr = np.array([])

for i in range(0, 153567):
    X_fft[i] = np.fft.fft(X[i])
    
    
for i in range(0, 153567):
    X_features[i, 16] = np.mean(X_fft[i])
    X_features[i, 17] = np.median(X_fft[i])
    X_features[i, 18] = np.std(X_fft[i])
    X_features[i, 19] = np.max(X_fft[i])
    X_features[i, 20] = np.var(X_fft[i])
    X_features[i, 21] = np.ptp(X_fft[i])
    X_features[i, 22] = np.percentile(X_fft[i], q = 10)
    X_features[i, 23] = np.percentile(X_fft[i], q = 25)
    X_features[i, 24] = np.percentile(X_fft[i], q = 50)
    X_features[i, 25] = np.percentile(X_fft[i], q = 75)
    X_features[i, 26] = np.percentile(X_fft[i], q = 90)
    X_features[i, 27] = scipy.stats.kurtosis(X_fft[i])
    X_features[i, 28] = scipy.stats.skew(X_fft[i])
    
    if (i <= 153566):
        X_features[i, 29] = np.correlate(X_fft[i], X_fft[i + 1]) #Corr of two consecutive bins
        
    if (i <= 153556):
        X_features[i, 30] = np.correlate(X_fft[i], X_fft[i + 10]) #Corr of 10 consecutive bins
        
X_fea = pd.DataFrame(X_features)
X_fea.to_csv("features_including_fft.csv")
"""





data = pd.read_csv("all_quakes.csv")
y = data.iloc[:, 4097].values

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
    for i in range(start, 129069, 36):
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



historys = []
scores = np.array([])
count = 0
mae = 3
while(True):
    count = count + 1
    model = Sequential()
    
    model.add(TimeDistributed(Dense(units = 256, activation = 'relu', kernel_initializer = 'uniform'), input_shape = (36, 31)))
    model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(units = 256, activation = 'relu', kernel_initializer = 'uniform')))
    model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(units = 128, activation = 'relu', kernel_initializer = 'uniform')))
    model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform')))
    model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform')))
    model.add(Dropout(.2))
    model.add(LSTM(units = 64, input_shape = (36, 64), kernel_initializer = 'uniform'))
    model.add(Dropout(.2))
    
    model.add(Dense(units = 1, kernel_initializer = 'uniform'))
    
    model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
    
    history = model.fit(data_in3d, y_data, batch_size = 10000, epochs = 15, validation_data = (data_in3dT, y_dataT))
    historys.append(history)
    
    y_pred = model.predict(data_in3d)
    mae = mean_absolute_error(y_data, y_pred)
    scores = np.append(scores, mae)


y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("y_pred_df.csv")
y_test_df = pd.DataFrame(y_data)
y_test_df.to_csv("y_test_df.csv")

model.save("overfittedv1.h5")
model = load_model("featureRNN_v1.h5")

model = load_model("actually_working.h5")

mean_absolute_error(y_data, y_pred)


indexsT = np.array([])
for start in range(129069, 129104):
    for i in range(start, 151821, 36):
        need_break = False
        for j in range(1, 16):
            if(i < quake_starts[j] and i + 35 > quake_starts[j]):
                need_break = True
        if(need_break):
            continue
        indexsT = np.append(indexsT, i)


indexsT = indexsT.astype(np.int32)
# shuffles indexes
np.random.shuffle(indexsT)

y_dataT = np.zeros((indexsT.size, 1))
all_indexsT = np.zeros((indexsT.size, 36))
for i in range(0, indexsT.size):
    y_dataT[i] = y[indexsT[i]+35]
    all_indexsT[i] = np.arange(indexsT[i], indexsT[i]+36)


all_indexsT.resize((indexsT.size, 36))
all_indexsT = all_indexsT.astype(np.int32)
    

data_in3dT = np.zeros((indexsT.size, 36, 31))

for i in range(0, indexsT.size):
    for j in range(0, 36):
        data_in3dT[i,j,:] = X_features[all_indexsT[i,j]]


y_predT = model.predict(data_in3dT)
mae = mean_absolute_error(y_dataT, y_predT)
print(mae)

plt.plot(y_dataT[:1000])
plt.plot(y_predT[:1000])


for i in range(0, 18):
    plt.plot(historys[i])


x1 = model.layers[-1]
x2 = model.layers[-2]
x3 = model.layers[-3]
x4 = model.layers[-4]
x5 = model.layers[-5]
x6 = model.layers[-6]
x7 = model.layers[-7]

d1 = Dropout(.8)
d2 = Dropout(.8)
d3 = Dropout(.8)
d4 = Dropout(.8)
d5 = Dropout(.8)
d6 = Dropout(.8)
d7 = Dropout(.8)


x = d7(x7.output)
x = x6(x)
x = d6(x)
x = x5(x)
x = d5(x)
x = x4(x)
x = d4(x)
x = x3(x)
x = d3(x)
x = x2(x)
x = d2(x)

x1 = x1(x)

model2 = Model(input = model.input, output = x1)








seq = pd.read_csv("sample_submission.csv")
seq_data = seq.iloc[:, 0].values
seq_data = seq_data.astype(str)



full_data = np.zeros((seq_data.size, 36, 31))


for k in range(seq_data.size):
    if(k%100 == 0):
        print(k)
    file = 'split/' + seq_data[k] + '.csv'
    data = pd.read_csv(file)
    data = data.iloc[:,:].values
    data_bins = np.zeros((36, 4096))
    for j in range(0, 36):
        data_bins[j] = np.reshape(data[j*4096:(j*4096)+4096], (4096))
        
    data_bins = np.absolute(data_bins)

    X_features = np.zeros((36, 31))

    for i in range(0, 36):
        X_features[i, 0] = np.mean(data_bins[i])
        X_features[i, 1] = np.median(data_bins[i])
        X_features[i, 2] = np.std(data_bins[i])
        X_features[i, 3] = np.max(data_bins[i])
        X_features[i, 4] = np.var(data_bins[i])
        X_features[i, 5] = np.ptp(data_bins[i])
        X_features[i, 6] = np.percentile(data_bins[i], q = 10)
        X_features[i, 7] = np.percentile(data_bins[i], q = 25)
        X_features[i, 8] = np.percentile(data_bins[i], q = 50)
        X_features[i, 9] = np.percentile(data_bins[i], q = 75)
        X_features[i, 10] = np.percentile(data_bins[i], q = 90)
        X_features[i, 11] = scipy.stats.entropy(data_bins[i])
        X_features[i, 12] = scipy.stats.kurtosis(data_bins[i])
        X_features[i, 13] = scipy.stats.skew(data_bins[i])
        
        if (i < 35):
            X_features[i, 14] = np.correlate(data_bins[i], data_bins[i + 1]) #Corr of two consecutive bins
            
        if (i < 26):
            X_features[i, 15] = np.correlate(data_bins[i], data_bins[i + 10]) #Corr of 10 consecutive bins
    
    X_fft = np.zeros((36 ,4096))
    fftArr = np.array([])
    
    for i in range(0, 36):
        X_fft[i] = np.fft.fft(data_bins[i])
    
    for i in range(0, 36):
        X_features[i, 16] = np.mean(X_fft[i])
        X_features[i, 17] = np.median(X_fft[i])
        X_features[i, 18] = np.std(X_fft[i])
        X_features[i, 19] = np.max(X_fft[i])
        X_features[i, 20] = np.var(X_fft[i])
        X_features[i, 21] = np.ptp(X_fft[i])
        X_features[i, 22] = np.percentile(X_fft[i], q = 10)
        X_features[i, 23] = np.percentile(X_fft[i], q = 25)
        X_features[i, 24] = np.percentile(X_fft[i], q = 50)
        X_features[i, 25] = np.percentile(X_fft[i], q = 75)
        X_features[i, 26] = np.percentile(X_fft[i], q = 90)
        X_features[i, 27] = scipy.stats.kurtosis(X_fft[i])
        X_features[i, 28] = scipy.stats.skew(X_fft[i])
        
        if (i < 35):
            X_features[i, 29] = np.correlate(X_fft[i], X_fft[i + 1]) #Corr of two consecutive bins
            
        if (i < 26):
            X_features[i, 30] = np.correlate(X_fft[i], X_fft[i + 10]) #Corr of 10 consecutive bins
        
    X_features = sc_X.transform(X_features)
    full_data[k] = X_features

np.save("all_data", full_data)

allData = np.load("all_data.npy")



