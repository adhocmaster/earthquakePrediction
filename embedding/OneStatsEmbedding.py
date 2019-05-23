import numpy as np
import logging
from scipy import stats
from data_analysis.library.Bin import Bin
from data_analysis.library.BinProcessor import BinProcessor
from embedding.Embedding import Embedding
from embedding.SourceCardinality import SourceCardinality
from embedding.Stats import Stats
import pandas as pd

class OneStatsEmbedding(Embedding):

    def __init__(self, scaler = None):

        self.numberOfFeatures = 15 + 6 * 27 + 2 + 15 + 3 * 27
        self.scaler = scaler
        self.stats = Stats()
        super(OneStatsEmbedding, self).__init__(sourceCardinality = SourceCardinality.MULTI)
        pass
    
    def fromBins(self, bins: Bin):

        # 1. get all data & scale it using the scaler
        data = []
        ttfs = []
        for aBin in bins:
            data.extend(aBin.data)
            ttfs.append(aBin.ttf)
        
        data = np.array(data).reshape(-1, 1)

        if self.scaler is not None:
            data = self.scaler.transform( data )
        # 2. conduct stats on the data

        with np.errstate(invalid='ignore'):
            embedding = self.stats.getBasicStatsList(data) #15 #maybe this function should use series
            dataSeries = pd.Series(data.flatten())
            embedding.extend(self.stats.getTrendStatsList(dataSeries)) # 6 * 27
            embedding.extend(self.stats.getLinearSeasonalityStatsList(data, True)) # 2
            embedding.extend(self.stats.getFirstOrderSeasonalityStatsList(dataSeries)) # 3 * 27
            # embedding.extend(self.stats.getTTFDiffStatsList(ttfs)) # 15

        #print (f"embedding length: {len(embedding)}" )

        # 3. return stats
        return np.array(embedding)

    
