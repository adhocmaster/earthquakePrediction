import numpy as np
import logging, dill, fnmatch, os
from embedding.EmbeddingCache import EmbeddingCache

class EmbeddingIO:

    def __init__(self):
        """
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        self.sourceSSD = '/home/exx/muktadir/data/train.csv'
        self.sourceHDD = '/home/exx/muktadir/data/train.csv'
        self.destFolderSSD = '/home/exx/muktadir/data/'
        self.destFolderHDD = '/home/exx/muktadir/data/'

        """
        self.sourceSSD = 'C:/earthquake/train.csv'
        self.sourceHDD = 'F:/myProjects/cmps242/earthquake/data/train.csv'
        self.destFolderSSD = 'C:/earthquake/'
        self.destFolderHDD = 'F:/myProjects/cmps242/earthquake/data/'
        self.destFolder = self.destFolderSSD

        pass

    def save(self, anEm, emType):

        fname = self.getFileName(anEm.embeddingId, emType)
        # print( fname)
        with open(fname, 'wb') as outfile:
            dill.dump(anEm, outfile)

        pass


    def getFileName(self, embeddingId, emType):
        return self.getFolder(emType) + self.getRelativeFileName(embeddingId, emType)

    
    def getFolder(self, emType):
        return self.destFolder + emType + '-embedding/'

    def getRelativeFileName(self, embeddingId, emType):
        return 'em_' + str( embeddingId ) + '.dill'
        
    def readById(self, embeddingId, emType):

        fname = self.getFileName(embeddingId, emType)
        return self.read(fname)


    def read(self, fname):

        with open(fname, 'rb') as f:
            out = dill.load(f)
        return out