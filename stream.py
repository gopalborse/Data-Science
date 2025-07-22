import numpy as np
import joblib
import streamlit as st

# Load the trained model and scaler
model = joblib.load("model_scaled.pkl")
scale = joblib.load("scaled.pkl")

# Streamlit app title
st.title("Diabetes Prediction App")
st.write("Enter your medical details to check your diabetic status")

# Sidebar input fields
st.sidebar.header("Your Medical Report")

preg = st.sidebar.number_input("Pregnancies", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
plas = st.sidebar.number_input("Plasma Glucose", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
pres = st.sidebar.number_input("Blood Pressure", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
skin = st.sidebar.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
test = st.sidebar.number_input("Insulin", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
mass = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
pedi = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
age = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Prepare input data
input_data = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
scaled_input = scale.transform(input_data)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(scaled_input)
    st.success(f"prediction:{prediction[0]}")
    #if prediction[0] == 1:
       # st.success("You may be diabetic. Please consult a doctor.")
   # else:
      #  st.success("You are likely not diabetic.")

