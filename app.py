import streamlit as st
import pandas as pd

st.title("Diet Meal Planning Optimisation using Evolutionary Programming")

data = pd.read_csv("Food_and_Nutrition__.csv")

st.subheader("Food and Nutrition Dataset")
st.dataframe(data)
