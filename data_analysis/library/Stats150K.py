import numpy as np

class Stats150K:

    def __init__(self):
        pass

    def createFromDf(self, df, windowSize=150_000, stopAfter = 0, addBinNoToDf = False, dontSaveToDisk = False):

        nextId = 0
        start = nextId * windowSize
        nextId += 1
        end = start + 150_000

        while end <= len(df) and (stopAfter == 0 or stopAfter >= nextId):

            nextDf = df[start:end]

            print( f"size of next df {len(nextDf)}, start {start}, end {end}")

            start = nextId * windowSize
            nextId += 1
            end = start + 150_000

        pass

    def resample(self, df, start, size):
        return df
