import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor

class RandomForestQuantileRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, n_estimators=10,max_depth=4,min_samples_leaf=1000):
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self,X_train,y_train):
        self.reg = ExtraTreesRegressor(min_samples_leaf=self.min_samples_leaf,
                                         n_jobs=-1, 
                                         verbose=True,
                                         bootstrap= False,
                                         max_depth=self.max_depth,
                                         n_estimators=self.n_estimators,random_state=2020)
        self.reg.fit(X_train, y_train)
        
        # We sort the training data per target data
        t = pd.DataFrame({"target":y_train})
        xt = pd.DataFrame(X_train)
        xt.columns = np.arange(len(xt.columns))
        d = pd.concat([xt,t],axis=1)
        d = d.sort_values("target")
        lendata = len(pd.DataFrame(X_train).iloc[0])
        self.X_train = d[np.arange(lendata)]
        self.y_train = d["target"].values

        est_len = len(self.reg.estimators_)
        obs_len = len(y_train)

        bins=40
        self.miny=.9*np.min(y_train)
        self.maxy=1.2*np.max(y_train)
        self.therange = np.linspace(self.miny,self.maxy,bins)
        intw = (self.maxy-self.miny)/bins

        def get_vector_cdf(estimator,X_train,y_train):
            obs_node = estimator.apply(X_train)
            w = {}
            idx_sort = np.argsort(obs_node)
            sorted_records_array = obs_node[idx_sort]
            uniques, idx_start, counts = np.unique(sorted_records_array, return_counts=True,return_index=True)
            res = np.split(idx_sort, idx_start[1:])
            for i in np.arange(len(uniques)):
                node=uniques[i]
                w[node]=10*pd.Series(np.cumsum(np.histogram(y_train[res[i]],
                    range=(self.miny,self.maxy),bins=40,density=True)[0]))
            return pd.DataFrame(w).T

        self.vectors_cdf_df = [get_vector_cdf(estimator,self.X_train,self.y_train) for estimator in self.reg.estimators_]
        
        
    def predict(self,X):
        est_len = len(self.reg.estimators_)

        def predict_quantile(row):
            total_vector = np.concatenate([ self.vectors[i][node_test] for i,node_test in enumerate(row) ])
            return self.y_train[total_vector]

        histos_test = []
        for i,estimator in enumerate(self.reg.estimators_):
            nodes_test = estimator.apply(X)
            histos = self.vectors_cdf_df[i].loc[nodes_test]
            histos_test.append(histos.values)
        res = 0
        for h in histos_test:
            res += h
        preds_df = pd.DataFrame(res/est_len)

        perc_df = pd.DataFrame()
        perc_df["10"] = self.therange[40-(preds_df>.1).T.sum()]
        perc_df["50"] = self.therange[40-(preds_df>.5).T.sum()]
        perc_df["90"] = self.therange[40-(preds_df>.9).T.sum()]

        return perc_df