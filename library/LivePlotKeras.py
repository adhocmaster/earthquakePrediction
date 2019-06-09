import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

import seaborn as sns
sns.set(style="darkgrid")

class LivePlotKeras(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure(figsize=(20, 10))
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('mean_squared_error'))
        self.val_losses.append(logs.get('val_mean_squared_error'))
        self.i += 1
        
        clear_output(wait=True)
        self.fig = plt.figure(figsize=(20, 10))
        plt.plot(self.x, self.losses, label="train")
        plt.plot(self.x, self.val_losses, label="validation")
        plt.legend()
        plt.show()

        