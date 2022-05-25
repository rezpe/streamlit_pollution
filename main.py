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

st.sidebar.header("Time Series Forecasting with ML models")

st.sidebar.title("1. Data")

with st.sidebar.expander("What is this ?"):
    st.write("""In this section, you will be able 
    in the future to load your own data in order to
    perform the predictions and then
    specify the date and target columns. For now, it 
    is predefined""")

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

limit = st.sidebar.slider('Limit (to reduce processing)', 0, len(df), 1000)
df=df[-limit:]

# Feature Selection
feat_selected = []

st.sidebar.title("2. Predictors")

with st.sidebar.expander("What is this ?"):
    st.write("""In this section, you can 
    select the signal periodic function
    so your model learns the seasonality.
    You can also define lagged values which 
    are the values in the past used to 
    predict the future. The minimum lagged value 
    can be considered the horizon of 
    forecasting.""")

# Periods
periods = st.sidebar.text_input("Periods")
if periods != "":
    periods = map(float, periods.split(","))
    n = np.arange(len(df))

    for period in periods:
        df["c"+str(period)] = np.cos(2.0*n*np.pi/period)
        feat_selected.append("c"+str(period))
        df["s"+str(period)] = np.sin(2.0*n*np.pi/period)
        feat_selected.append("s"+str(period))

# Lagged Values
lags = st.sidebar.text_input("Lagged Values")
if lags != "":
    lags = map(int, lags.split(","))
    n = np.arange(len(df))

    for lag in lags:
        df[f"target - {str(lag)}"] = df[target].shift(lag)
        feat_selected.append(f"target - {str(lag)}")

df = df.dropna()

# Stop is no feature is selected
if len(feat_selected)==0:
    st.markdown("Please select at least a period of lagged value")
    st.stop()

# Building model selection

X = df[feat_selected]
y = df[target]

st.sidebar.title("3. Metrics")

with st.sidebar.expander("What is this ?"):
    st.write("""In this section, you can 
    select the training/test cut.""")

cut = st.sidebar.slider('Training - Test Cut', 0, len(df), int(4*len(df)/5))
X_train = X.iloc[:cut]
y_train = y.iloc[:cut]
X_test = X.iloc[cut:]
y_test = y.iloc[cut:]

st.sidebar.title("4. Models")
with st.sidebar.expander("What is this ?"):
    st.write("""In this section, you can 
    select the ML model and some of its 
    hyperparameters.""")

depth = st.sidebar.number_input("Depth", 3, 14, 4)
qreg = DecisionTreeQuantileRegressor(int(depth))

# Final Result

qreg.fit(X_train, y_train)
preds = qreg.predict(X_test)
rmse = np.round(np.mean((y_test.values - preds["50"])**2), 2)
bias = np.round(np.mean((y_test.values - preds["50"])), 2)

st.header("The metrics of your model are:")
st.metric(label="RMSE", value=rmse)
st.metric(label="BIAS", value=bias)


st.header("Select a point in the chart below to see the prediction at that point.")
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

sel_date = draw_df["date"].iloc[pt_sel]
low_index = np.max([0,pt_sel-100])
high_index = np.min([len(draw_df),pt_sel+100])
draw_df = draw_df.iloc[low_index:high_index]


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
).interactive()+\
    alt.Chart(pd.DataFrame({
  'Date': [sel_date]
})).mark_rule().encode(
  x='Date:T'
).interactive()
st.altair_chart(c,use_container_width=True)

