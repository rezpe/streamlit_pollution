import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeQuantileRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        
    def fit(self,X_train,y_train):
        self.reg = DecisionTreeRegressor(max_depth=self.max_depth,random_state=2020)
        self.reg.fit(X_train, y_train)
        obs_node = self.reg.apply(X_train)
        observations = {}
        for node in obs_node:
            observations[node]=np.percentile(y_train[obs_node==node],[10,50,90])
        self.observations_df = pd.DataFrame(observations).T

    def predict(self,X):
        pred_node = self.reg.apply(X)
        values = self.observations_df.loc[pred_node].values
        values_df = pd.DataFrame(values)
        values_df.columns = ["10","50","90"]

        return values_df