# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and dataframe
df = pd.read_csv("df.csv")
pipe = pickle.load(open("pipe.pkl", "rb"))

st.title("💻 Laptop Price Predictor")

# --- User Inputs ---
company = st.selectbox('Brand', df['Company'].unique())
lap_type = st.selectbox("Type", df['TypeName'].unique())
ram = st.selectbox("RAM (GB)", [2,4,6,8,12,16,24,32,64])
weight = st.number_input("Weight of the Laptop (kg)", min_value=0.5)
touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
ips = st.selectbox("IPS Display", ['No', 'Yes'])
screen_size = st.number_input('Screen Size (inches)', min_value=10.0)
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
     '2880x1800','2560x1600','2560x1440','2304x1440']
)
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# --- Prediction ---
if st.button('Predict Price'):
    # Convert Yes/No to 1/0
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Compute PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    # Prepare query
    query = np.array([company, lap_type, ram, weight,
                      touchscreen_val, ips_val, ppi,
                      cpu, hdd, ssd, gpu, os]).reshape(1, 12)

    # Predict price
    price = int(np.exp(pipe.predict(query)[0]))

    st.success(f"💰 Predicted Price: ₹ {price}")
