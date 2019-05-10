# 1. start from an earth quake and go backward. Join every n bins together
# 2. start anywhere and join a window of n bins till an earthquake
# 3. Add an especial seperator  for bin time diff. Fill with zeros? experiment.
import numpy as np
import logging, dill, fnmatch, os
from .Bin import Bin
from .BinIO import BinIO

class BinJoiner:
    
    def __init__(self):
        