import numpy as np
import logging, dill, fnmatch, os
from data_analysis.library.Bin import Bin
from data_analysis.library.BinIO import BinIO
from library.RegressionDataGenerator import RegressionDataGenerator
from embedding.BinEmbedding import *

class SingleBinDataGenerator(RegressionDataGenerator): 
    
    def __init__(self, binType='nor', embedding='bin', dim=(64,64), batch_size=32, n_channels=1, shuffle=True):
    
        self.binType = binType
        self.embedding = embedding
        self.binIO = BinIO()
        self.numRawBins = 153584

        # Make IDs here.
        list_IDs = self.getListIds()
        self.embedder = self.getEmbedder()
    
        super(SingleBinDataGenerator, self).__init__(list_IDs, batch_size, dim, n_channels, shuffle)
        pass

    
    def getListIds(self):

        if self.embedding == 'bin':
            return list(range(1, self.numRawBins +1))
    

    def getEmbedder(self):
        
       if self.embedding == 'bin':
           return BinEmbedding(4096)


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
            X[i,] = self.embedder.fromBin(aBin)

            # Store class
            y[i] = aBin.ttf

        return X, y
    
    
