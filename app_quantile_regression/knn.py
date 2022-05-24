import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from scipy import stats

from sklearn.neighbors import KNeighborsRegressor

class QuantileKNN():
    
    def __init__(self, n_neighbors = 50):
        self.n_neighbors = n_neighbors
        
        #keep estimator in memory 
        self.reg = None
        
    def fit(self,X_train,y_train):
        self.neigh=KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.neigh.fit(X_train,y_train)
        self.X_train=X_train
        self.y_train=y_train
        
    def predict(self,x):
        def get_quantiles(element,indices,array):

            quantiles = np.arange(1,100)/100.0
            temp=array[indices]
            
            dist = stats.norm(np.mean(temp),np.std(temp))
            quant=[]
            for quantile in quantiles :
                quant.append(dist.ppf(quantile))
            
            return quant

        predictions_gbr=[]
        for element in tqdm(x):
            indices=self.neigh.kneighbors([element], return_distance=False)
            predictions_gbr.append(get_quantiles(element,indices,self.y_train))

        return predictions_gbr