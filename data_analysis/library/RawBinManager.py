import numpy as np
import pandas as pd
import collections
import logging, dill, fnmatch, os
from .Bin import Bin
from .BinIO import BinIO
from sklearn import preprocessing
from .Scalers import Scalers

class RawBinManager:

    def __init__(self, binType = 'r', makePositive = False, normalize = False, scale=False ):

        self.binIO = BinIO()
        self.scalers = Scalers()
        self.binType = binType
        # self.rawBinPrefix = binType + '_'
        # self.rawBinFolder = self.destFolderHDD + binType + '-bins/'

        self.curStatId = 0
        self.stats = {}
        self.makePositive = makePositive
        self.normalize = normalize
        self.scale = scale

        self.scaler = None
        self.normalizer = None

        pass

    def createRawBinsFromDf(self, df, stopAfter = 0, addBinNoToDf = False, dontSaveRawToDisk = False):

        if self.makePositive:
            df.acoustic_data = np.abs(df.acoustic_data)
            reshapedAcousticDataForPreprocessing = df.acoustic_data.values.reshape(-1, 1)
            if self.normalize:
                df['norm'] = self.scalers.getScaler('absNormalizer').transform(reshapedAcousticDataForPreprocessing)
                logging.warning('abs normalized df')

            if self.scale:
                df['scaled'] = self.scalers.getScaler('absScaler').transform(reshapedAcousticDataForPreprocessing)
                logging.warning('abs scaled df')
        else:
            reshapedAcousticDataForPreprocessing = df.acoustic_data.values.reshape(-1, 1)
            if self.normalize:
                df['norm'] = self.scalers.getScaler('normalizer').transform(reshapedAcousticDataForPreprocessing)
                logging.warning('normalized df')

            if self.scale:
                df['scaled'] = self.scalers.getScaler('scaler').transform(reshapedAcousticDataForPreprocessing)
                logging.warning('scaled df')


        if addBinNoToDf is True:
            df['binNo'] = np.zeros(len(df), dtype=np.int32)

        # 1. init stats
        self.initStatsForCurrentDf(df)

        # 2. Loop over bins
        nextId = 0
        index = -1
        nextBinDf, index = self.getNextBinDf(df, index)

        print( f"last index: {index} and number records in nextdf { nextBinDf.shape[0] } {nextBinDf.empty is False}" )
        while ( (nextBinDf.empty is False ) and (nextId <= stopAfter or stopAfter == 0) ):

            nextId = nextId + 1
            nextBin = self.convertDfIntoBinTuple(nextId, nextBinDf)
            if (nextId % 2000) == 0:
                    print( f'processed {nextId}th raw bin' )

            # 3. create bin stats
            self.addBinStats(nextBin)

            # 4. save bins
            if dontSaveRawToDisk is False:
                self.saveRawBin(nextBin, self.binType)

            if self.normalize:
                self.saveRawBin(self.getNormalBin(nextBin, nextBinDf), self.binType + 'nor')

            if self.scale:
                self.saveRawBin(self.getScaledBin(nextBin, nextBinDf), self.binType + 'scaled')

            # 5. augment df?
            if addBinNoToDf is True:
                self.addBinNoToDf(df, nextBinDf, nextId)


            # 6. next
            nextBinDf, index = self.getNextBinDf(df, index)

        if dontSaveRawToDisk is False:
            print(f'saved {nextId} bins to {self.rawBinFolder} folder')
        else:
            print(f'Processed {nextId} bins, but not saved.')

        pass


    def initStatsForCurrentDf(self, df):

        self.curStatId = len(df)
        self.stats[self.curStatId] = {}
        self.stats[self.curStatId]["earthquakeBinIds"] = []
        self.stats[self.curStatId]["sizeFrequencies"] = {}
        self.stats[self.curStatId]["binIdsBySize"] = {}

        pass



    def addBinStats(self, nextBin):

        sizeFrequencies = self.stats[self.curStatId]["sizeFrequencies"]
        binIdsBySize = self.stats[self.curStatId]["binIdsBySize"]

        sizeKey = len(nextBin.data)

        if sizeKey not in sizeFrequencies:
            sizeFrequencies[sizeKey] = 0
            binIdsBySize[sizeKey] = []

        sizeFrequencies[sizeKey] = sizeFrequencies[sizeKey] + 1
        binIdsBySize[sizeKey].append(nextBin.binId)

        pass


    def addBinNoToDf(self, df, nextBinDf, nextId):

        #print( nextBinDf.head(5) )

        for row in nextBinDf.itertuples(index = True):
            #print(f'adding binId {nextId} to row {row.Index}')
            df.loc[row.Index]['binNo'] = nextId

        pass


    def getNextBinDf(self, df, lastIndex = -1):
        """
        index is the end point of the last bin
        """

        start = lastIndex + 1

        if start >= df.shape[0]:
            return pd.DataFrame(), lastIndex

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
        """code smell: does earthquake calculations."""

        data = nextBinDf.acoustic_data.values
        ttf = nextBinDf.iloc[-1].time_to_failure

        quakeIndex = -1
        for i in range(1, len(data)):
            if nextBinDf.time_to_failure.iloc[i-1] - nextBinDf.time_to_failure.iloc[i] < -0.001:
                #negative value means ttf jumped. #todo confirm that this is correct. It can be incorrect.
                quakeIndex = i-1
                self.stats[self.curStatId]["earthquakeBinIds"].append( nextId )
                print( f'bin {nextId} has a quake at index {quakeIndex}' )
                break

        return Bin(binId = nextId,
                   ttf = ttf,
                   data = data,
                   quakeIndex = quakeIndex,
                   trIndexStart = nextBinDf.index[0]
                  )

    def getNormalBin(self, rawBin, rawBinDf):
        return Bin(binId = rawBin.binId,
                   ttf = rawBin.ttf,
                   data = rawBinDf.norm.values,
                   quakeIndex = rawBin.quakeIndex,
                   trIndexStart = rawBin.trIndexStart
                  )

    def getScaledBin(self, rawBin, rawBinDf):
        return Bin(binId = rawBin.binId,
             ttf = rawBin.ttf,
             data = rawBinDf.scaled.values,
             quakeIndex = rawBin.quakeIndex,
             trIndexStart = rawBin.trIndexStart
            )


    def saveRawBin(self, nextBin, binType ):
        self.binIO.saveBin( nextBin, binType )
        pass


    def readRawBinById(self, binIdn, binType):
        return self.binIO.readBinById(binId, binType)

    def countRawBin(self, fname):
        return self.binIO.countBin(self.binType)

    def readRawBins(self, fromId, size):
        return self.binIO.readBins(fromId, size, self.binType)
