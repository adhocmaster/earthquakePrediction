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
        self.embedding = OneStatsEmbedding(scaler)
        self.dim = (binsPerEmbedding, self.embedding.numberOfFeatures)
        super(CNNStatsEmbedding, self).__init__(sourceCardinality = SourceCardinality.MULTI)
        pass

    def fromBins(self, bins: Bin):

        # 1. get all data & scale it using the scaler
        data = []
        # ttfs = []
        for aBin in bins:
            binStats = self.embedding.fromUnnormalizedNumpyData(aBin.data)
            data.append(binStats)
        
        return np.array(data)
        
        