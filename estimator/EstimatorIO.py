import numpy as np
import logging, dill, fnmatch, os
from datetime import datetime
from keras.models import load_model
from keras.models import model_from_json

class EstimatorIO:

    def __init__(self, engine='keras'):
        """
        self.sourceHDD = 'F:/myProjects/tfKeras/UCSC/CMPS242/earthquake/estimator/cached/'
        self.sourceHDD = '/home/exx/muktadir/data/'

        """
        self.sourceHDD = 'F:/myProjects/tfKeras/UCSC/CMPS242/earthquake/estimator/cached/'
        self.destFolder = self.sourceHDD

        self.engine = engine

        pass

    def save(self, model, name):

        if self.engine == 'keras':
            self.saveKeras(model, name)
            return

        fname = self.getFileName(name)
        # print( fname)
        with open(fname, 'wb') as outfile:
            dill.dump(model, outfile)
        pass

    def saveKeras(self, model, name):
        # 1. Save Json
        # 2. Save weight
        # 3. Save Train Model
        # 4. save Test

        trainModelPath, testModelPath, jsonPath, weightPath = self.getJsonAndWeightPaths(name)
        json = model.to_json()

        with open(jsonPath, "w") as jF:
            jF.write(json)
        
        model.save_weights(weightPath)
        model.save(trainModelPath)

        # create a test model and save

        testModel = model_from_json(json) # Must not have used batch_input_shape
        testModel.load_weights(weightPath)
        testModel.compile(loss = model.loss, optimizer = model.optimizer, metrics = model.metrics)

        testModel.save(testModelPath)

        pass

    def getJsonAndWeightPaths(self, name):
        return self.getKerasTrainModelPath(name),  self.getKerasTestModelPath(name), self.getKerasDefPath(name), self.getKerasWeightPath(name)

    def getKerasTrainModelPath(self, name):
        return self.getFolder() + name + '-model-train.h5'

    def getKerasTestModelPath(self, name):
        return self.getFolder() + name + '-model-test.h5'

    def getKerasDefPath(self, name):
        return self.getFolder() + name + '-def.json'

    def getKerasWeightPath(self, name):
        return self.getFolder() + name + '-weight.h5'

    def getFileName(self, name):
        return self.getFolder() + self.getRelativeFileName(name)

    
    def getFolder(self):
        return self.destFolder

    def getRelativeFileName(self, name):
        return name + datetime.today().strftime('-%Y-%m-%d') + '.dill'
        
    def readByName(self, name):

        fname = self.getFileName(name)
        return self.read(fname)


    def read(self, fname):

        with open(fname, 'rb') as f:
            out = dill.load(f)
        return out

    
    def loadUnCompiledKerasModelWithWeights(self, name):

        with open(self.getKerasDefPath(name), 'r') as f:
            model = model_from_json(f.read())
            model.load_weights(self.getKerasWeightPath(name))
            return model


    def loadTestKerasModel(self, name):
        return load_model(self.getKerasTestModelPath(name))


    def loadTrainKerasModel(self, name):
        return load_model(self.getKerasTrainModelPath(name))
