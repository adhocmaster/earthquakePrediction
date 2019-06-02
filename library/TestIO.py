import numpy as np
import logging, dill, fnmatch, os
from embedding.EmbeddingIO import EmbeddingIO

class TestIO:
    
    def __init__(self, embeddingType = 'one-stats-test'):
        """
        
        self.sourceSSD = '/home/exx/muktadir/data/train.csv'
        self.sourceHDD = '/home/exx/muktadir/data/train.csv'
        self.destFolderSSD = '/home/exx/muktadir/data/'
        self.destFolderHDD = '/home/exx/muktadir/data/'

        """
        self.sourceSSD = 'C:/earthquake/test/'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/test/'
        self.destFolderSSD = 'C:/earthquake/test/one'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        self.destFolder = self.destFolderSSD
        self.sourceFolder = self.sourceSSD

        self.io = EmbeddingIO()
        self.embeddingType = embeddingType

        pass
    
    def save(self, anEm):
        self.io.save(anEm, self.embeddingType)

    def readById(self, embeddingId):
        return self.io.readById(embeddingId, self.embeddingType)