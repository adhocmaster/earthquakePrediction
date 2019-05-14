import numpy as np
import logging, dill, fnmatch, os
from data_analysis.library.Bin import Bin
from data_analysis.library.BinIO import BinIO
from .RegressionDataGenerator import RegressionDataGenerator

class EarthQuakeDataGenerator(RegressionDataGenerator): 
    
    def __init__(self, binType='r', batch_size=32, dim=(32,32,32), n_channels=1, shuffle=True):
    
        # Make IDs here.
    
        super(EarthQuakeDataGenerator, self).__init__(list_IDs, batch_size, dim, n_channels, shuffle)
        
        self.binType = binType
        self.binIO = BinIO()
        
        pass
    
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # read the bin
            aBin = self.binIO.readBinById(ID, self.binType)
            
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = np.load('data/' + ID + '_output.npy')

        return X, y
    
    
