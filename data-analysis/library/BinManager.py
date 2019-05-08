import numpy as np
import pandas as pd
import collections
import logging, dill, fnmatch, os
from .Bin import Bin

class BinManager:
    
    def __init__(self):
        
        """
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        """
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        self.sourceSSD = '/home/exx/muktadir/data/train.csv'
        self.sourceHDD = '/home/exx/muktadir/data/train.csv'
        self.destFolderSSD = '/home/exx/muktadir/data/'
        self.destFolderHDD = '/home/exx/muktadir/data/'
        
        self.rawBinPrefix = 'r_'
        self.rawBinFolder = self.destFolderHDD + 'raw-bins/'
        
        pass
    
    def createRawBinsFromDf(self, df, stopAfter = 0):
        
        nextId = 0
        index = -1
        nextBinDf, index = self.getNextBinDf(df, index)
        
        print( f"last index: {index} and number records in nextdf { nextBinDf.shape[0] } {nextBinDf.empty is False}" )
        while ( (nextBinDf.empty is False ) and (nextId <= stopAfter or stopAfter == 0) ):
        
            nextId = nextId + 1
            nextBin = self.convertDfIntoBinTuple(nextId, nextBinDf)

            self.saveRawBin(nextBin)   
            
            if (nextId % 10000) == 0:
                print( f'saved {nextId}th raw bin' )
            
            nextBinDf, index = self.getNextBinDf(df, index)
            
        print(f'saved {nextId} bins to {self.rawBinFolder} folder')
        
        pass        
    
    
    def getNextBinDf(self, df, lastIndex = -1):
        """
        index is the end point of the last bin
        """
        
        start = lastIndex + 1
        
        if start >= df.shape[0]:
            return pd.Dataframe(), lastIndex
        
        end = start + 4094
        
        while (end < df.shape[0]):
            
            if (end + 1) == df.shape[0]:
                break
                
            diff = df.time_to_failure[end] - df.time_to_failure[end+1]

            if diff > 0.00001:
                break

            end = end + 1
        
        return df[start:end+1], end
    
    def convertDfIntoBinTuple(self, nextId, nextBinDf):
        
        data = nextBinDf.acoustic_data.values
        ttf = nextBinDf.iloc[-1].time_to_failure
        
        quakeIndex = -1
        for i in range(1, len(data)):
            if nextBinDf.time_to_failure.iloc[i-1] - nextBinDf.time_to_failure.iloc[i] < -0.001:
                #negative value means ttf jumped. #todo confirm that this is correct. It can be incorrect.
                quakeIndex = i-1
                break
        
        return Bin(binId = nextId, ttf = ttf, data = data, quakeIndex = quakeIndex)
    
    
    def saveRawBin(self, nextBin):
        
        fname = self.rawBinFolder + self.getRelativeRawFileName(nextBin.binId)
        with open(fname, 'wb') as outfile:
            dill.dump(nextBin, outfile)
            
        pass
    
    
        
    def getRelativeRawFileName(self, binId):
        return self.rawBinPrefix + 'bin_' + str( binId ) + '.dill'
        
    
    
    def readRawBinById(self, binId):
        
        fname = self.rawBinFolder + self.getRelativeRawFileName(binId)
        return self.readRawBin(fname)
    
        
    def readRawBin(self, fname):
        
        with open(fname, 'rb') as f:
            out = dill.load(f)
        
        return out
    
    def countRawBin(self, fname):
        
        #return len(fnmatch.filter(os.listdir(self.rawBinFolder), '*.dill'))
        return len(os.listdir(self.rawBinFolder))
    
    def readRawBins(self, fromId, size):
        
        bins = []
        for i in range(size):
            bins.append( self.readRawBinById(fromId + i) )
        
        return bins
        
        
        
        
        
        
        
        
        
    