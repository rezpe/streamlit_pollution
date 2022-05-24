import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K 

from sklearn.model_selection import train_test_split

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class MLPQuantile():
    
    def __init__(self):

        self.estimators = []
        
    def fit(self,X_train,y_train):
        
        def MLPmodel():
            model = Sequential()
            model.add(Dense(len(X_train[0]), input_dim=len(X_train[0]), activation=LeakyReLU(alpha=0.3)))
            model.add(Dense(int(len(X_train[0])/2), activation=LeakyReLU(alpha=0.3)))
            model.add(Dense(1, activation='linear'))
            return model
        
        print("training !")

        X_ttrain, X_val, y_ttrain, y_val = train_test_split(X_train,y_train,test_size=.05,random_state=2020)

        for q in [0.022750131948179195,0.15865525393145707,0.5,0.8413447460685429,0.9772498680518208]:
            print(f"Quantile: {q}")
            model = MLPmodel()
            optim=Adam(learning_rate=0.001)
            model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer=optim)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)
            history = model.fit(X_ttrain, y_ttrain, 
                                epochs=200, batch_size=200,  
                                verbose=1,callbacks=[es],
                                validation_data=(X_val,y_val))
            self.estimators.append(model)
        print("Done")
        
    def predict(self,X):
        predictions_gbr = []
        print("predicting")
        for reg in tqdm(self.estimators):
            predictions_gbr.append(reg.predict(X))
         
        total_pred={}
        for i in range(len(predictions_gbr)):
            total_pred[i]=predictions_gbr[i][:,0]
            
        total_df=pd.DataFrame(total_pred)

        def process_row(row):
            v = row.values
            dif_mean = np.abs(v-v[2])
            mu = v[2]
            s = np.mean([dif_mean[0]/2,dif_mean[1],dif_mean[3],dif_mean[4]/2])
            mi_norm = stats.norm(mu,s)
            quant=[]
            for quantile in np.arange(1,100)/100.0 :
                quant.append(mi_norm.ppf(quantile))
            return pd.Series(quant)
 
        total_df = total_df.apply(process_row,axis=1)
        
        return total_df.values
