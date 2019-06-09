import numpy as np
from scipy import stats
from .Bin import Bin
import seaborn as sns
sns.set(style="darkgrid")

class BinProcessor:
    
    def __init__(self):
        pass
    
    def getBinStats(self, aBin):
        
        scistats = stats.describe( aBin.data )   
        dic = {}
        dic['mean'] = scistats.mean
        dic['var'] = scistats.variance
        dic['median'] = np.median( aBin.data )
        dic['skewness'] = scistats.skewness
        dic['kurtosis'] = scistats.kurtosis
        dic['max'] = scistats.minmax[1]
        dic['min'] = scistats.minmax[0]
        dic['dif_max_min'] = dic['max'] - dic['min']
        dic['dif_median_mean'] = dic['median'] - dic['mean']
        
        return dic
    
    
    def makeDataPositive(self, aBin):
        
        data = np.abs(aBin.data)
        return self.updateData(aBin, data)
    
    def updateData(self, aBin, newData):
        return Bin(binId = aBin.binId, 
                   ttf = aBin.ttf, 
                   data = newData, 
                   quakeIndex = aBin.quakeIndex,
                   trIndexStart = aBin.trIndexStart
                  )
    
    def plot(self, aBin, ax=None):
        x = np.arange(len(aBin.data))
        sns.scatterplot(x, aBin.data, s=10, ax=ax,
                        estimator=None, label=f'{aBin.binId}-ttf-{aBin.ttf}')
    
    