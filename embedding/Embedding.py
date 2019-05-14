import numpy as np
from data_analysis.library.Bin import Bin

# Base class which all embedding classes need to implement
class Embedding:
    
    def fromBin(self, aBin: Bin):
        raise Exception(f"{type(self)} has not implemented fromBin")
