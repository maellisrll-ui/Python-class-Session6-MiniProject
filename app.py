app.py

# importing the DataFrames using st.cache_data

import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data():
  raw_df = pd.read_csv("raw_data.csv")
  clean_df = pd.read_csv("clean_data.csv")
  return raw_df, clean_df

raw_df, clean_df = load_data()


#  loading the model/scaler/encoder (so that they only load once per session instead of on every interaction)
@st.cache_resource
def load_model():
  model =joblib.load("best_model.pkl")
  scaler =joblib.load("scaler.pkl")
  encoder =joblib.load("label_encoder.pkl")

return model, scaler, encoder

model, scaler, encoder = load_model()
