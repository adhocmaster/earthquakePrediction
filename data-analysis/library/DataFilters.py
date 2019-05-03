import numpy as np
import pandas as pd
import collections
import logging

TimeSlice = collections.namedtuple('TimeSlice', 'ttf data')

class DataFilter:
    
    def __init__(self):
        
        self.featureSize = 150_000 #which is a chunk of 0.0375 seconds of seismic data (ordered in time), which is recorded at 4MHz, 
        # hence 150'000 data points, and the output is time remaining until the following lab earthquake, in seconds.
        
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
        
        pass
    
    
    def createChunkIterator(self, chunkSizeInM = 100, ttfDtype = np.float64  ):
        chunkSize = chunkSizeInM * 1000000
        return pd.read_csv(
            self.sourceSSD, 
            chunksize=chunkSize, 
            dtype = {'acoustic_data': np.int16, 'time_to_failure':ttfDtype } 
        )
    
    
    def loadCSVFromHDD(self, filename, ttfDtype = np.float64 ):
        location = self.destFolderHDD + filename
        return pd.read_csv(location, dtype = {'acoustic_data': np.int16, 'time_to_failure':ttfDtype } )
    
    def getPositionalDataInNP( self, df, start, step ):
        return df[start::step].values
    
    
    def getPositionalDataFromChunks( self, chunks:pd.DataFrame, start, step, ignore_index = True ):
        """Assumes that positions are preserved across chunks"""
        data = pd.DataFrame()
        for chunk in chunks:
            data = data.append( chunk[start::step], ignore_index = ignore_index )
        return data
    
            
    def getPositionalDataInNPFromChunks( self, chunks:pd.DataFrame, start, step ):
        """Assumes that positions are preserved across chunks"""
        dataList = []
        for chunk in chunks:
            dataList.append(chunk.values.tolist())
            
        return np.array(dataList)
    
    
    def saveDF(self, df, filename, index = False):
        df.to_csv( self.destFolderHDD + filename, index = index, chunksize = 10000 )
        pass
    
    
    def savePositionalDFFromChunks( self, chunks:pd.DataFrame, start, step, ignore_index = True, rename=True  ):
        df = self.getPositionalDataFromChunks(chunks, start, step, ignore_index)
        if rename:
            df.columns = ['acoustic', 'ttf']
        filename = 'every_' + str(step) + '_from_' + str(start) + '.csv'
        self.saveDF(df, filename)
        pass
    
    def getBins( self, df ):
        """ TODO: Fix this. Bean boundary can be anywhere and two corner cases. diff is big ~ 0.001 or negative (after an earth quake)"""
        curTime = -1
        data = []
        tempSlice = []
        for row in df.itertuples(index = False):
            
            if curTime != row.time_to_failure:
                if curTime > -1:
                    #save it
                    print(f"appending {curTime} with {len(tempSlice)} data points")
                    data.append( TimeSlice(ttf=curTime, data= np.array(tempSlice) ) )
                    
                tempSlice = []
                curTime = row.time_to_failure
                
            tempSlice.append(row.acoustic_data)
        
        return data
    
    def printBinBoundary(self, df, binNo):
        # TODO: fix each packet is supposed to have 4096 samples
        start = 4096 * binNo - 10
        for i in range(20):
            start = start + 1
            if start in df.index:
                diff = df.time_to_failure[start-1] - df.time_to_failure[start]
                if diff < 0.00001:
                    print( f" {start-1}, {start}: {diff}" )
                else:
                    logging.warning( f" {start-1}, {start}: {diff}" )
        pass
    
    def getBin(self, df, binNo):
        """ TODO: This method won't work for bins too far or after an earthquake"""
        start = 4096 * (binNo - 1)
        samples = []
        #fix start if it's not beanNo 1
        if binNo > 1:
            diff = df.time_to_failure[start-1] - df.time_to_failure[start]
            while diff < 0.00001:
                start = start - 1
                diff = df.time_to_failure[start-1] - df.time_to_failure[start]

        diff = df.time_to_failure[start+4094] - df.time_to_failure[start+4095]
        if diff < 0.00001:
            samples = df[start:start+4096]
        else:
            samples = df[start:start+4095]
        
        return samples
    
    def getBinStats(self, df, binNo):
        binDf = self.getBin(df, binNo)
        dic = {}
        dic['mean'] = binDf.time_to_failure.mean()
        dic['var'] = binDf.time_to_failure.var()
        dic['median'] = binDf.time_to_failure.median()
        dic['max'] = binDf.time_to_failure.max()
        dic['min'] = binDf.time_to_failure.min()
        dic['dif_max_min'] = dic['max'] - dic['min']
        dic['dif_median_mean'] = dic['median'] - dic['mean']
        
        return dic
        
                
        
        
    
  