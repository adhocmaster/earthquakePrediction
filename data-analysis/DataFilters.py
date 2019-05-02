import numpy as np
import pandas as pd

class DataFilter:
    
    def __init__(self):
        self.source1 = 'C:/earthquake/train.csv'
        self.source2 = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolder1 = 'C:/earthquake/'
        self.destFolder2 = 'F:/myProjects/cmps242/earthquake/data/'
        pass
    
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
        df.to_csv( self.destFolder2 + filename )
        pass
    
    def savePositionalDFFromChunks( self, chunks:pd.DataFrame, start, step ):
        df = self.getPositionalDataFromChunks(chunks, start, step)
        filename = 'every_' + str(start) + '_from_' + str(step) + '.csv'
        self.saveDF(df, filename)
        pass