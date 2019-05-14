import numpy as np
from data_analysis.library.Bin import Bin
from .Embedding import Embedding

class BinEmbedding(Embedding):
    
    def __init__(self, binSize = 4096 ):
        
        self.binSize = 4096
        self.rowDim = 64
        self.colDim = int( self.binSize / self.rowDim )
        
        if self.binSize % self.rowDim != 0:
            raise Exception(f"{binSize} is not divisible by {self.rowDim}")
        pass
    
    def fromBin(self, aBin: Bin):
        
        curBinSize = len(aBin.data)
        
        if curBinSize < 4096:
            aBin = self.inflateBin(aBin, 4096)
        else if curBinSize > 4096:
            aBin = self.reduceBinWithQuake(aBin, 4096)
        
        return aBin.data.reshape([self.rowDim,self.colDim])
            
        