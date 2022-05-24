import streamlit as st

import numpy as np
import pandas as pd

import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from streamlit_plotly_events import plotly_events

from app_quantile_regression.lgba import TotalLGBQuantile
from app_quantile_regression.fastqrf import RandomForestQuantileRegressor
from app_quantile_regression.qdt import DecisionTreeQuantileRegressor


st.set_page_config(layout="wide")

st.sidebar.markdown("## Select input data")
df=pd.read_csv("electricity2.csv")


cols = df.columns
datefield = st.sidebar.selectbox('Date variable',list(cols))
try:
    df[datefield]=pd.to_datetime(df[datefield])
    df.sort_values(datefield,inplace=True)
except:
    st.stop()

target = st.sidebar.selectbox('Target variable',list(cols),1)
if df[target].dtype.kind!="f":
    st.markdown("Please select appropriate target")
    st.stop()

st.sidebar.write('Target:', target )

exogenous = st.sidebar.multiselect('Select exogenous variables',
                                list(set(cols)-set([target,datefield])),
                                list(set(cols)-set([target,datefield])))

other_time_series = st.sidebar.multiselect('Other time Series',
                                list(set(cols)-set([target,datefield])),
                                list(set(cols)-set([target,datefield])))

# Condition to continue
cond = False
for v in df[exogenous].dtypes.values:
    if str(v)=="object":
        cond=True
if cond:
    st.stop() 

cond = False
for v in df[other_time_series].dtypes.values:
    if str(v)=="object":
        cond=True
if cond:
    st.stop() 


# Feature Selection
feat_selected = []

# Periods
periods = st.sidebar.text_input("Periods")
if periods != "":
    periods = map(float, periods.split(" "))
    n = np.arange(len(df))

    for period in periods:
        df["c"+str(period)] = np.cos(2.0*n*np.pi/period)
        feat_selected.append("c"+str(period))
        df["s"+str(period)] = np.sin(2.0*n*np.pi/period)
        feat_selected.append("s"+str(period))

# Exogenous features
if len(exogenous)>0:
    st.sidebar.markdown("# Exogenous Variables")
    cal_features = []
    for cal_feat in exogenous:
        cal_features.append(st.sidebar.checkbox(cal_feat, True))

    feat_selected += list(exogenous[cal_features])

st.sidebar.markdown("Lagged features")

for col_name in other_time_series+[target]:
    lagt = st.sidebar.text_input(f"{col_name} lags")
    if lagt != "":
        lagt = map(int, lagt.split(" "))
        for l in lagt:
            df[f"{col_name} - "+str(l)] = df[col_name].shift(l)
            feat_selected.append(f"{col_name} - "+str(l))

df = df.dropna()

# Stop is no feature is selected
if len(feat_selected)==0:
    st.markdown("Please select at least a predictor")
    st.stop()

# Building model selection

X = df[feat_selected]
y = df[target]

cut = st.sidebar.slider('Training - Test Cut', 0, len(df), int(4*len(df)/5))
X_train = X.iloc[:cut]
y_train = y.iloc[:cut]
X_test = X.iloc[cut:]
y_test = y.iloc[cut:]

depth = st.sidebar.number_input("Depth", 3, 14, 4)
qreg = DecisionTreeQuantileRegressor(int(depth))

# Final Result

qreg.fit(X_train, y_train)
preds = qreg.predict(X_test)
rmse = np.round(np.mean((y_test.values - preds["50"])**2), 2)
bias = np.round(np.mean((y_test.values - preds["50"])), 2)

st.metric(label="RMSE", value=rmse)
st.metric(label="BIAS", value=bias)

fig = px.line(data_frame=df.iloc[cut:],y=target,x=datefield,
            title="Target Variable Evolution",
            labels={ 
                target: f"Target Variable:{target}",  datefield: f"Date: {datefield}", 
            })
selected_points = plotly_events(fig)
try:
    pt_sel = selected_points[0]["pointIndex"]
except:
    st.write("Select point in chart")
    st.stop()

# Chart Draw
draw_df = pd.DataFrame()
draw_df["10"]=preds["10"].values
draw_df["50"]=preds["50"].values
draw_df["90"]=preds["90"].values
draw_df["target"]=y_test.values
draw_df["date"]=df[datefield].iloc[cut:].values
draw_df = draw_df.iloc[pt_sel-100:pt_sel+100]

c = alt.Chart(draw_df).mark_line().encode(
    x="date:T",
    y="10",
    color=alt.value("orange")
).interactive()+\
alt.Chart(draw_df).mark_line().encode(
    x="date:T",
    y="50",
    color=alt.value("green")
).interactive()+\
alt.Chart(draw_df).mark_line().encode(
    x="date:T",
    y="90",
    color=alt.value("orange")
).interactive()+\
alt.Chart(draw_df).mark_line().encode(
    x="date:T",
    y="target",
    color=alt.value("blue"),
    tooltip="date"
).interactive()
st.altair_chart(c,use_container_width=True)

