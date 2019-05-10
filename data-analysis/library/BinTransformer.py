import numpy as np
import logging, dill, fnmatch, os
from .Bin import Bin
from .BinManager import BinManager
from .BinProcessor import BinProcessor

class BinTransformer:
    
    def __init__(self):
        
        self.binManager = BinManager()
        self.binProcessor = BinProcessor()
        
        self.numRawBins = 153584
        
        self.destFolderSSD = self.binManager.destFolderSSD
        self.destFolderHDD = self.binManager.destFolderHDD
        
        self.positiveBinPrefix = 'p_'
        self.positiveBinFolder = self.destFolderHDD + 'positive-bins/'
                
        pass
    
    
    # For positive bins
    def createPositiveBins(self, fromId = 1, toId = 0 ):
        
        if toId == 0:
            toId = self.numRawBins 
        
        for binId in range(fromId, toId + 1):
            if (binId % 2000) == 0:
                print( f'processed {binId}th positive bin' ) 
            positiveBin = self.binProcessor.makeDataPositive( self.binManager.readRawBinById(binId) )
            self.savePositiveBin( positiveBin )
        
        pass
    
    
    def savePositiveBin(self, positiveBin):
        
        fname = self.positiveBinFolder + self.getRelativePositiveFileName(positiveBin.binId)
        with open(fname, 'wb') as outfile:
            dill.dump(positiveBin, outfile)
            
        pass
    
    
    def getRelativePositiveFileName(self, binId):
        return self.positiveBinPrefix + 'bin_' + str( binId ) + '.dill'
        
    
    def readPositiveBinById(self, binId):
        
        fname = self.positiveBinFolder + self.getRelativePositiveFileName(binId)
        return self.readPositiveBin(fname)
    
        
    def readPositiveBin(self, fname):
        
        with open(fname, 'rb') as f:
            out = dill.load(f)
        
        return out
    
    def countPositiveBin(self, fname):
        
        return len(os.listdir(self.positiveBinFolder))
    
    def readPositiveBins(self, fromId, size):
        
        bins = []
        for i in range(size):
            bins.append( self.readPositiveBinById(fromId + i) )
        
        return bins
            
            
        
        
