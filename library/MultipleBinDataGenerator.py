import numpy as np
import logging, dill, fnmatch, os, math
from data_analysis.library.Bin import Bin
from data_analysis.library.BinIO import BinIO
from data_analysis.library.Scalers import Scalers
from library.RegressionDataGenerator import RegressionDataGenerator
from embedding.OneStatsEmbedding import *
from embedding.MultipleBinEmbeddingType import *

class MultipleBinDataGenerator(RegressionDataGenerator): 
    
    def __init__(self, binType='pos', embedding=MultipleBinEmbeddingType.ONE_STATS, startBinId = 1, windowSize = 36, stride = 36, list_IDs = None, numBins = 153584,  batch_size=32, n_channels=1, shuffle=False):
    
        self.binType = binType
        self.embedding = embedding
        self.binIO = BinIO()
        self.startBinId = startBinId
        self.numBins = numBins
        self.scalers = Scalers()
        self.windowSize = windowSize
        self.stride = stride

        # Make IDs here.
        if list_IDs is None:
            list_IDs = self.getListIds()

        self.embedder = self.getEmbedder()

        if self.stride > self.windowSize:
            logging.warning( f"stride is greater than windowSize" )

        logging.warning(f"shuffling: {shuffle}")
    
        super(MultipleBinDataGenerator, self).__init__(list_IDs, batch_size, dim=(self.embedder.numberOfFeatures,), shuffle = shuffle)

        pass

    
    def getListIds(self):

        return list(range(self.startBinId, self.numBins +1))
    

    def getEmbedder(self):
        
        if self.embedding == MultipleBinEmbeddingType.ONE_STATS:
            if self.binType == 'nor':
                return OneStatsEmbedding( self.scalers.getScaler('scaler') )
            elif self.binType == 'pos':
                return OneStatsEmbedding( self.scalers.getScaler('absScaler') )

    def getNumberOfBatches(self):
        
        if self.stride >= self.windowSize:
            return math.floor(self.numBins / (self.stride * self.batch_size))
        else:
            return math.floor(( self.numBins + 1 - self.windowSize) / (self.stride * self.batch_size) ) # TODO verfiy this equation. 

    def __len__(self):
        return self.getNumberOfBatches()

    def __getitem__(self, batchIndex):

        'Generate one batch of data'

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size)

        #print( X.shape )
        #print(self.dim)

        sampleStartId = batchIndex * self.batch_size * self.stride + 1

        for i in range( self.batch_size ):
            X[i,], y[i] = self.getEmbeddingAndOutput(sampleStartId)
            sampleStartId += self.stride

        print(X.shape)
        return X, y


    def getEmbeddingAndOutput(self, startBinId ):
        
        endBinId = startBinId + self.windowSize
        bins = []
        for binId in range( startBinId, endBinId ):
            try:
                bins.append( self.binIO.readBinById(binId, self.binType) )
            except Exception as e:
                logging.warning( f"Batch bin exception. Might be safe to continue. {e}")

        lastBin = bins[-1]

        # Generate data
        features = self.embedder.fromBins(bins)

        #print(features.shape)

        return features, lastBin.ttf


    
