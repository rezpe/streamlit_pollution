import lightgbm as lgb
import numpy as np
import math
import pandas as pd
from scipy import stats

class TotalLGBQuantile():
    
    def __init__(self,n_estimators,max_depth):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.quantiles=[0.1,0.5,0.9]
        self.estimators = []
        
    def fit(self,X_train,y_train):
        for q in self.quantiles:
            reg = lgb.LGBMRegressor(n_estimators=self.n_estimators,
                                    objective= 'quantile',
                                    loss="quantile",
                                    alpha=q,
                                    random_state=2020,
                                   max_depth=self.max_depth)
                                
            reg.fit(X_train, y_train)
            self.estimators.append(reg)
        print("Done")
        
    def predict(self,X):
        preds_total=[]
        for estimator in self.estimators:
            preds=estimator.predict(X)
            preds_total.append(preds)
        
        preds_df = pd.DataFrame(preds_total).T
        preds_df.columns= ["10","50","90"]

        return preds_df
