import numpy as np
import pandas as pd
import json

from mlinsights.mlmodel import QuantileLinearRegression

class TotalLinearQuantile():
    
    def __init__(self):
        self.estimators = []
        self.quantiles=[0.1,0.5,0.9]     
        
    def fit(self,X_train,y_train):
        for q in self.quantiles:
            qreg= QuantileLinearRegression(quantile=q)
            qreg.fit(X_train,y_train)
            self.estimators.append(qreg)
             
    def predict(self,X):
        preds_total=[]
        for estimator in self.estimators:
            preds=estimator.predict(X)
            preds_total.append(preds)
        
        preds_df = pd.DataFrame(preds_total).T
        preds_df.columns= ["10","50","90"]

        return preds_df

