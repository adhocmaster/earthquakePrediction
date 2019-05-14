import numpy as np
import logging, dill
from .Bin import Bin
from .BinManager import BinManager
from .BinProcessor import BinProcessor
from .BinIO import BinIO

class PositiveBinManager:
    
    def __init__(self):
        
        self.binManager = BinManager()
        self.binProcessor = BinProcessor()
        self.binIO = BinIO()
        
        self.numRawBins = 153584
        
        self.positiveBinType = 'pos'
                
        pass
    
    
    # For positive bins
    def createPositiveBins(self, fromId = 1, toId = 0 ):
        
        """ It makes all the acoustic data from raw bins positive """
        
        if toId == 0:
            toId = self.numRawBins 
        
        for binId in range(fromId, toId + 1):
            if (binId % 2000) == 0:
                print( f'processed {binId}th positive bin' ) 
            positiveBin = self.binProcessor.makeDataPositive( self.binManager.readRawBinById(binId) )
            self.binIO.saveBin( positiveBin, self.positiveBinType )
        
        pass
    
    
    def readPositiveBinById(self, binId):
        
        return self.binIO.readBinById(binId, self.positiveBinType)
    
    
    def countPositiveBin(self):
        
        return self.binIO.countBin(self.positiveBinType)
    
    def readPositiveBins(self, fromId, size):
        
        return self.binIO.readBins(fromId, size, self.positiveBinType)
            
            
        
        
