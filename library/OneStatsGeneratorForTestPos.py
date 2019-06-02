from library.TestIO import TestIO
from os.path import dirname, basename, isfile
import glob
import pandas as pd
import numpy as np
import math, re
from embedding.OneStatsEmbedding import *
from embedding.EmbeddingCache import EmbeddingCacheTest
from data_analysis.library.Scalers import Scalers

class OneStatsGeneratorForTestPos:
    def __init__(self, windowSize = 200):
        self.embeddingType = 'one-stats-test'
        self.io = TestIO(self.embeddingType)
        self.scalers = Scalers()
        self.windowSize = windowSize
        self.lastEmbeddingId = 0
        self.numEmbeddings = 0
        self.numberOfEmbeddingPerFile = math.ceil((150_000 - 36 * 4096) / windowSize)
        self.embedder = OneStatsEmbedding( self.scalers.getScaler('absScaler') ) # positive scaler
        pass
    
    def generateEmbeddings(self):

        csvPaths = glob.glob(dirname(self.io.sourceFolder)+"/*.csv")

        i = 0
        for path in csvPaths:
            self.createEmbeddingsFromPath(path)
            i += 1
            if i % 100 == 0:
                print(f"processed {i} files")
                print(f"generated {self.lastEmbeddingId} embeddings")
        
        
        print(f"generated {self.lastEmbeddingId} embeddings")
    
    def createEmbeddingsFromPath(self, path):
        df = pd.read_csv(
                    path, 
                    dtype = {'acoustic_data': np.int16} 
                )
        
        df.acoustic_data = df.acoustic_data.abs() # converting to positive vals.
        
        start = 0
        for _ in range(self.numberOfEmbeddingPerFile):
            end = start + 4096
            binDf = df[start: end]
            # create embedding
            self.createEmbedddingFromBinDf(binDf)
            start = end

        pass


    def createEmbedddingFromBinDf(self, binDf):

        features = self.embedder.fromUnnormalizedDfData(binDf)
        self.lastEmbeddingId += 1
        embedding = EmbeddingCacheTest(embeddingId=self.lastEmbeddingId, type=self.embeddingType, features = features)

        self.io.save(embedding)

        pass
    
    def getBatch(self,  batchNo, batchSize=16):

        start = (batchNo - 1) * batchSize + 1
        end = start + 16

        batchList = []

        for embeddingId in range(start, end):
            try:
                embedding = self.io.readById(embeddingId)
                batchList.append(embedding.features)
            
            except Exception as e:
                logging.warning(f'encountered exception while reading embedding #{embeddingId}: {e}. Sliently progressing')
                break
            
        pass

        return np.array(batchList)
    
    def batches(self, batchSize = 16):

        # numBatches = math.ceil(self.numEmbeddings / 16)

        # for i in range(numBatches):
        #     yield self.getBatch(i+1, batchSize)
        
        # pass
        i = 0
        while True:
            i = i + 1
            data = self.getBatch(i, batchSize)
            if len(data) == 0:
                break
            yield data

    def batchesByFile(self):

        csvPaths = glob.glob(dirname(self.io.sourceFolder)+"/*.csv")
        i = 0
        for path in csvPaths:
            i = i + 1
            data = self.getBatch(i, self.numberOfEmbeddingPerFile)
            if len(data) == 0:
                break
            yield self.getTestName(path), data
        
    def getTestName(self, path):
        print(path)
        return re.findall(r'.*[\/\\]([a-zA-Z0-9_]+)\.csv$', path)[0]
        

    
