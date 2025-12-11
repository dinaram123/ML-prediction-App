import streamlit as st
import numpy as np 
import pickle
model = pickle.load(open("model.pkl", "rb"))
st.title("ML prediction App Using streamlit")
st.write("Enter age and BMI to predict insurance charges.")
age = st.number_input("Enter Age", min_value=1, max_value=100, step=1)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, step=0.1)
if st.button("predict"):
    features = np.array([[age, bmi]])
    prediction = model.predict(features)[0]
    st.success(f"predicted Insurance Charges:₹{prediction:.2f}")
