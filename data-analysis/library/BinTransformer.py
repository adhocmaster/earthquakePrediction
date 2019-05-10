import numpy as np
from .Bin import Bin
from .BinManager import BinManager
from .BinProcessor import BinProcessor

class BinTransformer:
    
    def __init__(self):
        
        self.binManager = BinManager()
        self.BinProcessor = BinProcessor()
        
        self.numRawBins = 153584
        
        self.destFolderSSD = self.binManager.destFolderSSD
        self.destFolderHDD = self.binManager.destFolderHDD
        
        self.positiveBinPrefix = 'r_'
        self.positiveBinFolder = self.destFolderHDD + 'positive-bins/'
        
        pass
    
    
    # For positive bins
    def createPositiveBins(self):
        
        for binId in range(1, self.numRawBins + 1):
            positiveBin = binProcessor.makeDataPositive( binManager.readRawBinById(binId) )
            self.savePositiveBin( positiveBin )
        
        pass
    
    
    def savePositiveBin(self, positiveBin):
        
        fname = self.positiveBinFolder + self.getRelativePositiveFileName(positiveBin.binId)
        with open(fname, 'wb') as outfile:
            dill.dump(positiveBin, outfile)
            
        pass
    
    
    def getRelativePositiveFileName(self, binId):
        return self.positiveBinPrefix + 'bin_' + str( binId ) + '.dill'
            
            
        
        
