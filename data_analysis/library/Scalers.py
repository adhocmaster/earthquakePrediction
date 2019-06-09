from sklearn import preprocessing
import numpy as np
import logging, dill

class Scalers:
    def __init__(self):
        #self.scalerFolder = '/home/exx/muktadir/earthquakePrediction/scalers/'
        self.scalerFolder = './scalers/'
        pass

    def createScalers(self, df):
        reshapedAcousticDataForPreprocessing = df.acoustic_data.values.reshape(-1, 1)
        absValues = np.abs( reshapedAcousticDataForPreprocessing )

        self.normalizer = preprocessing.MinMaxScaler((0,5)).fit(reshapedAcousticDataForPreprocessing)
        with open(self.scalerFolder + 'normalizer', 'wb') as outfile:
            dill.dump(self.normalizer, outfile)

        self.scaler = preprocessing.RobustScaler().fit(reshapedAcousticDataForPreprocessing)
        with open(self.scalerFolder + 'scaler', 'wb') as outfile:
            dill.dump(self.scaler, outfile)

        self.absNormalizer = preprocessing.MinMaxScaler((0,5)).fit(absValues)
        with open(self.scalerFolder + 'absNormalizer', 'wb') as outfile:
            dill.dump(self.absNormalizer, outfile)

        self.absScaler = preprocessing.RobustScaler().fit(absValues)
        with open(self.scalerFolder + 'absScaler', 'wb') as outfile:
            dill.dump(self.absScaler, outfile)

        pass

    def getScaler(self, name):
        fname = self.scalerFolder + name
        with open(fname, 'rb') as f:
            out = dill.load(f)
        return out
