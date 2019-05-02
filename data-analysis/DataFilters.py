import numpy as np
import pandas as pd

class DataFilter:
    
    def __init__(self):
        pass
    
    def getPositionalDataInNP( self, df, start, step ):
        return df[start::step].values
    
    def getPositionalDataFromChunks( self, chunks:pd.DataFrame, start, step ):
        """Assumes that positions are preserved across chunks"""
        data = pd.DataFrame()
        for chunk in chunks:
            data.append( chunk[start::step] )
        return data
            
    def getPositionalDataInNPFromChunks( self, chunks:pd.DataFrame, start, step ):
        """Assumes that positions are preserved across chunks"""
        dataList = []
        for chunk in chunks:
            dataList.append(chunk.values.tolist())
            
        return np.array(dataList)