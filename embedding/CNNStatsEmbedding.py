import numpy as np
from data_analysis.library.Bin import Bin
from embedding.Embedding import Embedding
from embedding.SourceCardinality import SourceCardinality
from embedding.OneStatsEmbedding import OneStatsEmbedding

class CNNStatsEmbedding(Embedding):

    """similar features in a column. We will run 2-D CNNN with 1-D kernel"""

    def __init__(self, scaler = None, binsPerEmbedding = 36):

        self.type = 'cnn-stats'
        self.scaler = scaler
        self.binsPerEmbedding = binsPerEmbedding
        self.embedding = OneStatsEmbedding(scaler)
        self.dim = (binsPerEmbedding, self.embedding.numberOfFeatures, 1)
        super(CNNStatsEmbedding, self).__init__(sourceCardinality = SourceCardinality.MULTI)
        self.numberOfFeatures = self.embedding.numberOfFeatures
        pass

    def fromBins(self, bins: Bin):

        # 1. get all data & scale it using the scaler
        data = []
        # ttfs = []
        for aBin in bins:
            binStats = self.embedding.fromUnnormalizedNumpyData(aBin.data)
            data.append(binStats)
        
        return np.array(data)
    
    def fromBinsDf(self, df):
        start = 0
        data = []
        for _ in range(self.binsPerEmbedding):
            end = start + 4096
            binStats = self.embedding.fromUnnormalizedNumpyData(df[start: end])
            data.append(binStats)
            start = end
        
        return np.array(data)

        
        