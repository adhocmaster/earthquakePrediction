import numpy as np
import logging, dill
from .Bin import Bin
from .BinProcessor import BinProcessor
from .BinIO import BinIO

class BinNormalizer:

    def __init__(self, min = -5500, max = 5500):

        self.binProcessor = BinProcessor()
        self.binIO = BinIO()

        self.min = min
        self.max = max
        self.range = max - min

        self.numBins = 153584

        self.toBinType = 'nor'

        pass

    
    def normByMinMax(self, aBin: Bin):
        
        data = (aBin.data - self.min) / self.range
        return self.binProcessor.updateData(aBin, data)
    
    
    def createNormalizedBins(self, binType = 'r', fromId = 1, toId = 0 ):
        
        """ It makes all the acoustic data from raw bins positive """
        
        if toId == 0:
            toId = self.numBins 
        
        for binId in range(fromId, toId + 1):
            if (binId % 2000) == 0:
                print( f'processed {binId}th bin' ) 
            
            fromBin = self.binIO.readBinById(binId, binType);
            toBin = self.normByMinMax(fromBin)

            self.binIO.saveBin( toBin, self.toBinType )
        
        pass
    