import numpy as np
from data_analysis.library.Bin import Bin
from .Embedding import Embedding

class BinEmbedding(Embedding):
    
    def __init__(self, binSize = 4096 ):
        pass
    
    def fromBin(self, aBin: Bin):
        
        curBinSize = len(aBin.data)
        
        if curBinSize < 4096:
            aBin = self.inflateBin(aBin, 4096)
        else if curBinSize > 4096:
            aBin = self.reduceBinWithQuake(aBin, 4096)
        
        
            
        