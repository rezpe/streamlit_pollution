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
#from app_quantile_regression.linear import TotalLinearQuantile
from app_quantile_regression.fastqrf import RandomForestQuantileRegressor
from app_quantile_regression.qdt import DecisionTreeQuantileRegressor

st.set_page_config(layout="wide")

"""## Session - Folder for the timeline
## st.session_state['kfolder'] stores the folder with  the session data
if 'kfolder' not in st.session_state:
    st.session_state['kfolder'] = "session_record"+datetime.now().strftime("%Y%m%d%H%M")
    os.mkdir(st.session_state['kfolder'])
    with open(st.session_state['kfolder']+"/session.json","w") as output:
        output.write("[]")
"""

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}
import matplotlib
matplotlib.rc('font', **font)

st.sidebar.markdown("""# Time Series Prediction Interpretability

This tool allows to train a probabilistic model and understand its predictions. The steps to use it are:
- Select data source and state its characteristics
- Select a train/test cut:
    - This will train the model
- Select a point to predict in the test set. For this:
    - Select a range in the first slider. You will select the point in this range
    - Select a point in the second slider.
- Click on predict to start prediction

""")

st.sidebar.markdown("## Select input data")
datas = st.sidebar.radio("Select input data",['Example: Electricity', 'Upload your own'],0)
df=None

if datas == 'Example: Electricity':
    df=pd.read_csv("electricity2.csv")
else:
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)


# Stop if no dataframe has been chosen
if df is None:
    st.stop()
cols = df.columns

datefield = st.sidebar.selectbox('Date variable',list(cols))
try:
    df[datefield]=pd.to_datetime(df[datefield])
    df.sort_values(datefield,inplace=True)
except:
    st.stop()

target = st.sidebar.selectbox('Target variable',list(cols))
if df[target].dtype.kind!="f":
    st.markdown("Please select appropriate target")
    st.stop()

st.sidebar.write('Target:', target )
threshold = st.sidebar.slider("Threshold",
                                value=float(df[target].mean()),
                                min_value=float(df[target].min()),
                                max_value=float(df[target].max()))

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

model = st.sidebar.selectbox(
    "Select Model", [ "boosted tree", "qrf","qdt"], 0)

if model == "linear":
    qreg = TotalLinearQuantile()
elif model == "boosted tree":
    depth = st.sidebar.number_input("Depth", 3, 14, 4)
    n_estimators = st.sidebar.selectbox(
        "Estimators", [50, 100, 200, 300, 500], 0)
    qreg = TotalLGBQuantile(int(n_estimators), int(depth))
elif model == "qrf":
    depth = st.sidebar.number_input("Depth", 3, 14, 4)
    n_estimators = st.sidebar.selectbox(
        "Estimators", [50, 100, 200, 300, 500], 0)
    qreg = RandomForestQuantileRegressor(int(n_estimators), int(depth))
elif model == "qdt":
    depth = st.sidebar.number_input("Depth", 3, 14, 4)
    qreg = DecisionTreeQuantileRegressor(int(depth))

# Final Result

qreg.fit(X_train, y_train)
preds = qreg.predict(X_test)
rmse = np.round(np.mean((y_test.values - preds["50"])**2), 2)
bias = np.round(np.mean((y_test.values - preds["50"])), 2)

# Session recording
#session = json.load(open(st.session_state['kfolder']+"/session.json"))
"""session.append({
    "bias": bias,
    "rmse": rmse,
    "feat_selected": feat_selected,
    "model": model
})
json.dump(session, open(st.session_state['kfolder']+"/session.json", "w"))
session_df = pd.DataFrame(session)""" 

# Selecting "tabs"
navigation = st.radio("Select option",
                ["Evaluation metrics",
                "Specific Test Point",
                "Last point prediction",
                "Feature Importance"],
                0)

if navigation=="Evaluation metrics":

    # RMSE/BIAS/NLL
    st.metric(label="RMSE", value=rmse)
    st.metric(label="BIAS", value=bias)

    """ fig = plt.figure(figsize=(15, 5))
    for msel in session_df["model"].unique():
        msession_df=session_df[session_df["model"]==msel]
        plt.scatter(msession_df.index,msession_df["rmse"],label=msel)
    plt.title("Evolution of RMSE during experiment iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("RMSE Value")
    plt.legend()
    st.pyplot(fig)""" 


elif navigation=="Specific Test Point":

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

elif navigation=="Feature Importance":

    """reset=st.button("Reset timeline")
    if reset:
        json.dump([], open(st.session_state['kfolder']+"/session.json", "w"))

    total_features = np.concatenate([list(flist) for flist in session_df["feat_selected"].values])
    total_features = np.unique(total_features)

    all_difs = []
    for feat in total_features:
        dif = pd.DataFrame()
        dif["appear"]=session_df["feat_selected"].apply(lambda l: feat in l)-\
        session_df["feat_selected"].apply(lambda l: feat in l).shift(1)
        dif["appear"]=dif["appear"].shift(-1)
        dif["rmse_dif"]=session_df["rmse"]-session_df["rmse"].shift(-1)
        dif["var_appear"]=dif["appear"]*dif["rmse_dif"]
        dif=dif[dif["appear"]!=0]
        dif = dif.dropna()
        dif["feature"]=feat
        all_difs.append(dif)
    all_difs = pd.concat(all_difs)

    fig = px.strip(all_difs, x="feature", y="var_appear")
    st.plotly_chart(fig)


    #if model=="linear":
    #    fig = plt.figure(figsize=(15, 5))
    #    sns.barplot(y=feat_selected,x=qreg.estimators[2].coef_)
    #    st.pyplot(fig)"""

