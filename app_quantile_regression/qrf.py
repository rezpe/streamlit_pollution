import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy.fft import fft
import datetime
import json
import numpy as np
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import stats
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso, LinearRegression

class RandomForestQuantileRegressorSlow(BaseEstimator, RegressorMixin):
    
    def __init__(self, n_estimators=10,max_depth=4,min_samples_leaf=1000):
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self,X_train,y_train):
        self.reg = ExtraTreesRegressor(min_samples_leaf=self.min_samples_leaf,
                                         max_features="sqrt",
                                         n_jobs=-1, 
                                         verbose=True,
                                         bootstrap= False,
                                         max_depth=self.max_depth,
                                         n_estimators=self.n_estimators,random_state=2020)
        self.reg.fit(X_train, y_train)
        
        # We sort the training data per target data
        print("Ordering the Data")
        t = pd.DataFrame({"target":y_train.values})
        d = pd.concat([pd.DataFrame(X_train.values),t],axis=1)
        d = d.sort_values("target")
        self.X_train = d[np.arange(len(X_train.values[0]))]
        self.y_train = d["target"].values
        
        print("Buffering the vectors")

        quantiles = np.arange(1,10)/10.0 
        est_len = len(self.reg.estimators_)
        obs_len = len(self.y_train)

        def get_vector(estimator,X_train,y_train):
            obs_node = estimator.apply(X_train)
            w = {}
            idx_sort = np.argsort(obs_node)
            sorted_records_array = obs_node[idx_sort]
            uniques, idx_start, counts = np.unique(sorted_records_array, return_counts=True,return_index=True)
            res = np.split(idx_sort, idx_start[1:])
            for i in range(len(uniques)):
                node=uniques[i]
                w[node]=np.zeros(obs_len)
                w[node][res[i]] = 1/(counts[i]*est_len)
            return w
            
        self.vectors = [get_vector(estimator,self.X_train,self.y_train) for estimator in tqdm(self.reg.estimators_)]

        print("Done")
        
        
    def predict(self,X):
        def predict_quantile(x):
    
            total_vector = np.zeros(len(self.y_train))
            for i,estimator in enumerate(self.reg.estimators_):
                node_test = estimator.apply([x])[0]
                v = self.vectors[i][node_test]
                total_vector += v

            ws = np.cumsum(total_vector)

            pos = 0
            q_obs = []
            quantiles = [0.022750131948179195,0.15865525393145707,0.5,0.8413447460685429,0.9772498680518208]
            for i,v in enumerate(ws):
                if v>quantiles[pos]:
                    pos+=1
                    value = self.y_train[i]
                    q_obs.append(value)
                    if pos==len(quantiles):
                        break
            
            v = np.array(q_obs)
            dif_mean = v-v[2]
            mu = v[2]
            s = np.mean([-dif_mean[0]/2,-dif_mean[1],dif_mean[3],dif_mean[4]/2])
            
            mi_norm = stats.norm(mu,s)
            
            q_obs_dist = [mi_norm.ppf(quant) for quant in np.arange(1,100)/100.0]

            return q_obs_dist
            
        prob_pred=[]
        for x in tqdm(X.values):
            prob_pred.append(predict_quantile(x))
        
        return prob_pred