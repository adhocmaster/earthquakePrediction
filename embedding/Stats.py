import numpy as np
import logging
from scipy import stats
import pandas as pd
from data_analysis.library.Bin import Bin

from sklearn.linear_model import *

class Stats:

    def getBasicStatsList(self, data:np.ndarray):
        
        data = data[np.isfinite(data)]
        scistats = stats.describe( data )   
        embedding = []
        embedding.append(scistats.mean)
        embedding.append(scistats.variance)
        embedding.append(np.median( data ))
        embedding.append(scistats.skewness)
        embedding.append(scistats.kurtosis)
        embedding.append(scistats.minmax[1])
        embedding.append(scistats.minmax[0])
        embedding.append(scistats.minmax[1] - scistats.minmax[0])
        embedding.append(np.quantile(data, 0.99))
        embedding.append(np.quantile(data, 0.95))
        embedding.append(np.quantile(data, 0.90))
        embedding.append(np.quantile(data, 0.01))
        embedding.append(np.quantile(data, 0.05))
        embedding.append(np.quantile(data, 0.10))
        embedding.append(scistats.variance - scistats.mean)
        # 15 features upto var/mean
        
        return embedding
    

    def getTrendStatsList(self, x:pd.core.series.Series, windows = [5, 10, 20, 40, 100, 1000]):

        embedding = []
        
        for w in windows:
            x_roll_abs_mean = x.abs().rolling(w).mean().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_min = x.rolling(w).min().dropna().values
            x_roll_max = x.rolling(w).max().dropna().values
            
            embedding.append( x_roll_std.mean() )
            embedding.append( x_roll_std.std())
            embedding.append( x_roll_std.max())
            embedding.append( x_roll_std.min())
            embedding.append( np.quantile(x_roll_std, 0.01))
            embedding.append( np.quantile(x_roll_std, 0.05))
            embedding.append( np.quantile(x_roll_std, 0.10))
            embedding.append( np.quantile(x_roll_std, 0.95))
            embedding.append( np.quantile(x_roll_std, 0.99))
            
            embedding.append( x_roll_mean.mean())
            embedding.append( x_roll_mean.std())
            embedding.append( x_roll_mean.max())
            embedding.append( x_roll_mean.min())
            embedding.append( np.quantile(x_roll_mean, 0.05))
            embedding.append( np.quantile(x_roll_mean, 0.95))
            
            embedding.append( x_roll_abs_mean.mean())
            embedding.append( x_roll_abs_mean.std())
            embedding.append( np.quantile(x_roll_abs_mean, 0.05))
            embedding.append( np.quantile(x_roll_abs_mean, 0.95))
            
            embedding.append( x_roll_min.std())
            embedding.append( x_roll_min.max())
            embedding.append( np.quantile(x_roll_min, 0.05))
            embedding.append( np.quantile(x_roll_min, 0.95))

            embedding.append( x_roll_max.std())
            embedding.append( x_roll_max.min())
            embedding.append( np.quantile(x_roll_max, 0.05))
            embedding.append( np.quantile(x_roll_max, 0.95))
            # 27 features per loop
        
        # 6x27 = 162 features upto var/mean default
        return embedding
    
    def getLinearSeasonalityStatsList(self, arr, abs_values=False):

        embedding = []
        """Fit a univariate linear regression and return the coefficient."""
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)

        embedding.append( lr.coef_[0] )
        embedding.append( lr.intercept_ )

        return embedding
    
    def getFirstOrderSeasonalityStatsList(self, x:pd.core.series.Series):

        seasonalData = x.diff()
        embedding = self.getBasicStatsList(seasonalData.values) #15
        embedding.extend( self.getTrendStatsList(seasonalData, windows=[5,10, 20])) # 3 * 27
        return embedding

    
    def getTTFDiffStatsList(self, ttfs):

        return self.getBasicStatsList(ttfs)

