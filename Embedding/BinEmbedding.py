import numpy as np
import logging
from data_analysis.library.Bin import Bin
from data_analysis.library.BinProcessor import BinProcessor
from Embedding.Embedding import Embedding

class BinEmbedding(Embedding):
    
    def __init__(self, binSize = 4096 ):
        
        self.binProcessor = BinProcessor()

        self.binSize = 4096
        self.rowDim = 64
        self.colDim = int( self.binSize / self.rowDim )
        
        if self.binSize % self.rowDim != 0:
            raise Exception(f"{binSize} is not divisible by {self.rowDim}")
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



            
        