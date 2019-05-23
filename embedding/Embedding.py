import numpy as np
from data_analysis.library.Bin import Bin
from embedding.SourceCardinality import SourceCardinality

# Base class which all embedding classes need to implement
class Embedding:

    def __init__(self, sourceCardinality=SourceCardinality.SINGLE):
        self.sourceCardinality = sourceCardinality
        pass
    
    def fromBin(self, aBin: Bin):
        raise Exception(f"{type(self)} has not implemented fromBin")
    def fromBins(self, aBin: Bin):
        raise Exception(f"{type(self)} has not implemented fromBins")
