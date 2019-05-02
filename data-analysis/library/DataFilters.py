import numpy as np
import pandas as pd

class DataFilter:
    
    def __init__(self):
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        pass
    
    
    def createChunkIterator(self, chunkSizeInM = 100 ):
        chunkSize = chunkSizeInM * 1000000
        return pd.read_csv(
            self.sourceSSD, 
            chunksize=chunkSize, 
            dtype = {'acoustic_data': np.int16, 'time_to_failure':np.float64 } 
        )
    
    
    def getPositionalDataInNP( self, df, start, step ):
        return df[start::step].values
    
    
    def getPositionalDataFromChunks( self, chunks:pd.DataFrame, start, step ):
        """Assumes that positions are preserved across chunks"""
        data = pd.DataFrame()
        for chunk in chunks:
            data = data.append( chunk[start::step] )
            #print( 'chunk shape' + str( chunk.shape ) )
            #print( 'current shape' + str( data.shape ) )
        return data
    
            
    def getPositionalDataInNPFromChunks( self, chunks:pd.DataFrame, start, step ):
        """Assumes that positions are preserved across chunks"""
        dataList = []
        for chunk in chunks:
            dataList.append(chunk.values.tolist())
            
        return np.array(dataList)
    
    
    def saveDF(self, df, filename):
        df.to_csv( self.destFolderHDD + filename )
        pass
    
    
    def savePositionalDFFromChunks( self, chunks:pd.DataFrame, start, step ):
        df = self.getPositionalDataFromChunks(chunks, start, step)
        filename = 'every_' + str(step) + '_from_' + str(start) + '.csv'
        self.saveDF(df, filename)
        pass