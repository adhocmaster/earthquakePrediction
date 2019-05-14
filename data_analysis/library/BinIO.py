import numpy as np
import logging, dill, fnmatch, os
from .Bin import Bin

class BinIO:
    
    def __init__(self):
        """
        self.sourceSSD = '/home/exx/muktadir/data/train.csv'
        self.sourceHDD = '/home/exx/muktadir/data/train.csv'
        self.destFolderSSD = '/home/exx/muktadir/data/'
        self.destFolderHDD = '/home/exx/muktadir/data/'
        """
        
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        self.destFolder = self.destFolderSSD
        
        pass
    
    
    
    def saveBin(self, aBin, binType):
        
        fname = self.getBinFileName(aBin.binId, binType)
        # print( fname)
        with open(fname, 'wb') as outfile:
            dill.dump(aBin, outfile)
            
        pass
    
    
    def getBinFileName(self, binId, binType):
        return self.getBinFolder(binType) + self.getRelativeFileName(binId, binType)
    
    def getBinFolder(self, binType):
        return self.destFolder + binType + '-bins/'
        
    def getRelativeFileName(self, binId, binType):
        return binType + '_bin_' + str( binId ) + '.dill'
        
    
    def readBinById(self, binId, binType):
        
        fname = self.getBinFileName(binId, binType)
        return self.readBin(fname)
    
        
    def readBin(self, fname):
        
        with open(fname, 'rb') as f:
            out = dill.load(f)
        
        return out
    
    def countBin(self, binType):
        
        return len( os.listdir( self.getBinFolder(binType) ) )
    
    def readBins(self, fromId, size, binType):
        
        bins = []
        for i in range(size):
            bins.append( self.readBinById(fromId + i, binType) )
        
        return bins