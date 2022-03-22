import streamlit as st
import pandas as pd
import requests
import os

ip = os.environ['RETRAIN_IP']

st.set_page_config(layout="wide")
st.write("""
# Model Retraining Interface
""")

st.sidebar.header('User Input')
retraining_mode = st.sidebar.selectbox('Retraining Type',("all model","specific model"))

# Collects user input features into dataframe
if retraining_mode=='all model':
    mc_data = st.sidebar.file_uploader("Upload Machine Data", type=["csv"])
    ct_data = st.sidebar.file_uploader("Upload Chemical Tin Data", type=["csv"])
    st_data = st.sidebar.file_uploader("Upload Solder Thickness Data", type=["csv"])
    epoch = st.sidebar.number_input("epochs",step=1)

    if mc_data is not None:
        mc_data = pd.read_csv(mc_data)
        mc_data.to_csv("mc.csv",index=False)
        st.text('Machine Data Summary')
        st.dataframe(mc_data.describe())
        #minioClient.fput_object("dataset","mc.csv","mc.csv")
    if ct_data is not None:
        ct_data = pd.read_csv(ct_data)
        ct_data.to_csv("ct.csv",index=False)
        st.text('Chemical Tin Data Summary')
        st.dataframe(ct_data.describe())
        #minioClient.fput_object("dataset","ct.csv","ct.csv")
    if st_data is not None:
        st_data = pd.read_csv(st_data)
        st_data.to_csv("st.csv",index=False)
        st.text('Solder Thickness Data Summary')
        st.dataframe(st_data.describe())
        #minioClient.fput_object("dataset","st.csv","st.csv")

    if st.sidebar.button("Start Retrain"):
        url = 'http://nginx:81/retrain_model_all_streamlit'
        headers = {'Content-Type': 'application/json'}

        obj = {"epoch":int(epoch),
                "mc_path":"mc.csv",
                "ct_path":"ct.csv",
                "st_path":"st.csv"
                }
        x = requests.post(url, json = obj, headers=headers)
        st.info(f'Model Retraining Started, See the progress [here](http://{ip}:3000)')

elif retraining_mode=='specific model':
    model_name = st.sidebar.selectbox('Model Name',('chemical_tin','solder_thickness'))
    model_type = st.sidebar.selectbox('Model Type',(3,24,168))

    mc_data = st.sidebar.file_uploader("Upload Machine Data", type=["csv"])
    spc_data = st.sidebar.file_uploader("Upload Chemical Tin Data", type=["csv"]) if model_name=='chemical_tin' else st.sidebar.file_uploader("Upload Solder Thickness Data", type=["csv"])
    epoch = st.sidebar.number_input("epochs",step=1)

    if mc_data is not None:
        mc_data = pd.read_csv(mc_data)
        mc_data.to_csv("mc.csv",index=False)
        st.text('Machine Data Summary')
        st.dataframe(mc_data.describe())
        #minioClient.fput_object("dataset","mc.csv","mc.csv")
    if spc_data is not None:
        spc_data = pd.read_csv(spc_data)
        spc_data.to_csv("spc.csv",index=False)
        st.text(f'{model_name} Data Summary')
        st.dataframe(spc_data.describe())
        #minioClient.fput_object("dataset","spc.csv","spc.csv")

    if st.sidebar.button("Start Retrain"):
        url = 'http://nginx:81/retrain_model_streamlit'
        headers = {'Content-Type': 'application/json'}
        obj = {"model_name":model_name,
                "hours":model_type,
                "epoch":int(epoch),
                "mc_path":"mc.csv",
                "spc_path":"spc.csv"
                }
        x = requests.post(url, json = obj, headers=headers)
        st.info(f'Model Retraining Started, See the progress [here](http://{ip}:3000)')
        