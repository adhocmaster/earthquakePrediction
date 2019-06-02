import numpy as np
import logging, dill, fnmatch, os, math, gc
from data_analysis.library.Bin import Bin
from data_analysis.library.BinIO import BinIO
from data_analysis.library.Scalers import Scalers
from library.RegressionDataGenerator import RegressionDataGenerator
from embedding.OneStatsEmbedding import *
from embedding.CNNStatsEmbedding import *
from embedding.MultipleBinEmbeddingType import *
from embedding.EmbeddingCache import EmbeddingCache
from embedding.EmbeddingIO import EmbeddingIO

class MultipleBinDataGenerator(RegressionDataGenerator): 
    
    def __init__(self, binType='pos', embedding=MultipleBinEmbeddingType.ONE_STATS, 
                startBinId = 1, windowSize = 36, stride = 36, 
                list_IDs = None, numBins = 153584,  batch_size=32, 
                n_channels=1, shuffle=False):
    
        self.binType = binType
        self.embedding = embedding
        self.binIO = BinIO()
        self.startBinId = startBinId
        self.numBins = numBins
        self.scalers = Scalers()
        self.windowSize = windowSize
        self.stride = stride

        self.lastWindowBins = {}

        # Make IDs here.
        if list_IDs is None:
            list_IDs = self.getListIds()

        self.embedder = self.getEmbedder()
        self.embeddingIO = EmbeddingIO()

        if self.stride > self.windowSize:
            logging.warning( f"stride is greater than windowSize" )

        logging.warning(f"shuffling: {shuffle}")

        if embedding == MultipleBinEmbeddingType.ONE_STATS:
            self.dim = (self.embedder.numberOfFeatures)
        elif embedding == MultipleBinEmbeddingType.CNN_STATS:
            self.dim = self.embedder.dim
    
        super(MultipleBinDataGenerator, self).__init__(list_IDs, batch_size, dim=self.dim, shuffle = shuffle)

        pass

    
    def getListIds(self):

        return list(range(self.startBinId, self.numBins +1))
    

    def getEmbedder(self):
        
        if self.embedding == MultipleBinEmbeddingType.ONE_STATS:
            if self.binType == 'nor':
                return OneStatsEmbedding( self.scalers.getScaler('scaler') )
            elif self.binType == 'pos':
                return OneStatsEmbedding( self.scalers.getScaler('absScaler') )

        if self.embedding == MultipleBinEmbeddingType.CNN_STATS:
            if self.binType == 'nor':
                return CNNStatsEmbedding( self.scalers.getScaler('scaler'), binsPerEmbedding=self.windowSize )
            elif self.binType == 'pos':
                return CNNStatsEmbedding( self.scalers.getScaler('absScaler'), binsPerEmbedding=self.windowSize )

    def getNumberOfBatches(self):
        
        if self.stride >= self.windowSize:
            return math.floor(self.numBins / (self.stride * self.batch_size))
        else:
            return math.floor(( self.numBins + 1 - self.windowSize) / (self.stride * self.batch_size) ) # TODO verfiy this equation. 

    def __len__(self):
        return self.getNumberOfBatches()

    def __getitem__(self, batchIndex):

        'Generate one batch of data'

        if self.embedding == MultipleBinEmbeddingType.ONE_STATS:
            X = np.empty((self.batch_size, self.dim))
        if self.embedding == MultipleBinEmbeddingType.CNN_STATS:
            X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size)

        #print( X.shape )
        #print(self.dim)

        embeddingId = batchIndex * self.batch_size + 1

        try:

            for i in range( self.batch_size ):
                embeddingCache = self.embeddingIO.readById(embeddingId, self.embedder.type)
                # print(embeddingCache.features.shape)
                X[i,] = embeddingCache.features
                y[i] = embeddingCache.ttf
                embeddingId += 1

        except Exception as e:

            logging.warning(f"Batch exception{e}")

        #print(X.shape)
        return X, y

    # def __getitemFromBins__(self, batchIndex):

    #     'Generate one batch of data'

    #     X = np.empty((self.batch_size, self.dim))
    #     y = np.empty(self.batch_size)

    #     #print( X.shape )
    #     #print(self.dim)

    #     sampleStartId = batchIndex * self.batch_size * self.stride + 1

    #     for i in range( self.batch_size ):
    #         X[i,], y[i] = self.getEmbeddingAndOutput(sampleStartId)
    #         sampleStartId += self.stride

    #     #print(X.shape)
    #     return X, y


    def getEmbeddingAndOutput(self, startBinId ): #should cache the last window as there will be overlapping bins.
        
        endBinId = startBinId + self.windowSize
        bins = []
        for binId in range( startBinId, endBinId ):
            try:
                if binId in self.lastWindowBins:
                    bins.append(self.lastWindowBins[binId])
                else:
                    bins.append(self.binIO.readBinById(binId, self.binType))
            except Exception as e:
                logging.warning( f"Batch bin exception. Might be safe to continue. {e}")

        lastBin = bins[-1]
        #cache bins
        self.lastWindowBins = {}
        for aBin in bins:
            self.lastWindowBins[aBin.binId] = aBin

        # Generate data
        features = self.embedder.fromBins(bins)

        #print(features.shape)

        return features, lastBin.ttf
    
    def cacheEmbeddingByBatch(self, startEmbeddingId = 1, stopAfter =0):
        
        embeddingId = startEmbeddingId
        startBinId = (embeddingId-1) * self.stride + 1

        while startBinId + self.stride <= self.numBins and ( stopAfter == 0 or stopAfter >= startBinId ):
            # print(startBinId)
            features, ttf = self.getEmbeddingAndOutput(startBinId)
            embeddingCache = EmbeddingCache(embeddingId = embeddingId,
                                            firstBinId = startBinId,
                                            type = str(self.embedding.value) + '-w' + str(self.windowSize) + 's-' + str(self.stride),
                                            features = features,
                                            ttf = ttf)
            self.embeddingIO.save(embeddingCache, self.embedder.type)

            if embeddingId % 1000 == 0:
                logging.debug(f"cached {embeddingId} now collecting garbage")
                gc.collect()  # TODO do it in another thread
            startBinId += self.stride
            embeddingId += 1
            



    
