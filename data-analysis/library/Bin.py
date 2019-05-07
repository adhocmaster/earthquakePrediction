import numpy as np
import pandas as pd
import collections
import logging

Bin = collections.namedtuple( 'Bin', 'id ttf data quakeIndex' ) #quakeIndex -1 means no quake in this bin

class BinManager:
    
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
        
        self.rawBinPrefix = 'r_'
        self.rawBinFolder = self.destFolderHDD + 'raw-bins/'
        
        pass
    
    def createRawBinsFromDf(self, df, stopAfter = 0):
        
        nextId = 0
        index = -1
        nextBinDf, index = self.getNextBinDf(df, index)
        
        while (False != nextBinDf) and (nextId <= stopAfter or stopAfter == 0):
        
            nextId = nextId + 1
            nextBin = self.convertDfIntoBinTuple(nextId, nextBinDf)

            self.saveRawBin(nextBin)   
            
            if (nextId % 10000) == 0:
                print( f'saved {nextId}th raw bin' )
            
            nextBinDf, index = self.getNextBinDf(df, index)
        
        pass        
    
    
    def getNextBinDf(self, df, lastIndex = -1):
        """
        index is the end point of the last bin
        """
        
        start = lastIndex + 1
        
        if start >= df.shape[0]:
            return
        
        end = start + 4094
        
        while (end < df.shape[0]):
            
            if (end + 1) == df.shape[0]:
                break
                
            diff = df.time_to_failure[end] - df.time_to_failure[end+1]

            if diff > 0.00001:
                break

            end = end + 1
        
        return df[start:end+1]
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
    