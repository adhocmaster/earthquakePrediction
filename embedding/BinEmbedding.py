import numpy as np
import logging
from data_analysis.library.Bin import Bin
from embedding.Embedding import Embedding
from SourceCardinality import SourceCardinality

class BinEmbedding(Embedding):
    
    def __init__(self, binSize = 4096 ):
        
        self.binSize = binSize
        self.rowDim = 64
        self.colDim = int( self.binSize / self.rowDim )
        
        if self.binSize % self.rowDim != 0:
            logging.error(f"{binSize} is not divisible by {self.rowDim}")
            raise Exception(f"{binSize} is not divisible by {self.rowDim}")

        super(BinEmbedding, sourceCardinality = SourceCardinality.SINGLE)
        pass
    
    def fromBin(self, aBin: Bin):
        
        curBinSize = len(aBin.data)
        
        data = None

        if curBinSize < self.binSize:
            data = self.inflateBinData(aBin, self.binSize)
        elif curBinSize > self.binSize:
            data = self.reduceBinDataWithQuake(aBin, self.binSize)
        else:
            data = aBin.data
        
        return data.reshape([self.rowDim,self.colDim, 1])
    
    def inflateBinData(self, aBin, binSize):

        itemsToInflate = binSize - len(aBin.data)
        last = [aBin.data[-1]] * itemsToInflate
        return np.append( aBin.data, last )
    
    def reduceBinDataWithQuake(self, aBin, binSize):
        
        data = None
        logging.debug(f"bin {aBin.binId} has been reduced")

        if( aBin.quakeIndex >= binSize ): #take the second part
            logging.debug(f"bin {aBin.binId} has quakeIndex at {aBin.quakeIndex}")
            data = aBin.data[-binSize: len(aBin.data)]
        else:
            data = aBin.data[0: binSize]
        
        return data



            
        