# CKD Predictor App with Enhanced UI for Doctors
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from pathlib import Path

# App Configuration
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# Custom CSS (Blue Theme)
st.markdown("""
    <style>
        body {
            background-color: #f4f8fb;
        }
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #004080;
        }
        .section-header {
            font-size: 1.5rem;
            color: #007acc;
            margin-top: 2rem;
            border-bottom: 2px solid #007acc;
            padding-bottom: 0.5rem;
        }
        .login-box {
            background-color: #e3f2fd;
            padding: 2rem;
            border-radius: 10px;
            border: 1px solid #90caf9;
        }
        .stButton > button {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #005f99;
        }
    </style>
""", unsafe_allow_html=True)

# Load Model
def load_model():
    model_path = "ckd_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        X, y = make_classification(n_samples=400, n_features=24, n_classes=2)
        model = RandomForestClassifier()
        model.fit(X, y)
        return model

model = load_model()

# Hash Password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# File to store users
USER_DB = Path("users.csv")
if not USER_DB.exists():
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_DB, index=False)

# User Authentication
def signup(username, password):
    users = pd.read_csv(USER_DB)
    if username in users.username.values:
        return False, "Username already exists."
    new_user = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True, "Signup successful. Please log in."

def login(username, password):
    users = pd.read_csv(USER_DB)
    password_hash = hash_password(password)
    user = users[(users.username == username) & (users.password == password_hash)]
    return not user.empty

# Prediction Logic
def predict_ckd(input_data):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    return prediction, probability

# Main App Functionality
def show_predictor():
    st.markdown("<div class='main-header'>CKD Risk Prediction Tool</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Patient Medical Information</div>", unsafe_allow_html=True)

    st.write(f"Doctor: **{st.session_state['username']}**")

    patient_name = st.text_input("Patient Name")
    patient_gender = st.radio("Patient Gender", ["Male", "Female"])

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        blood_pressure = st.number_input("Blood Pressure", 50, 250, 120)
        specific_gravity = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
        sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
        red_blood_cells = st.radio("Red Blood Cells", ["normal", "abnormal"])
        pus_cell = st.radio("Pus Cell", ["normal", "abnormal"])
        pus_cell_clumps = st.radio("Pus Cell Clumps", ["present", "notpresent"])
        bacteria = st.radio("Bacteria", ["present", "notpresent"])
        blood_glucose = st.number_input("Blood Glucose Random", 50, 500, 120)
        blood_urea = st.number_input("Blood Urea", 10, 200, 30)
        serum_creatinine = st.number_input("Serum Creatinine", 0.1, 15.0, 1.5)

    with col2:
        sodium = st.number_input("Sodium", 120, 160, 138)
        potassium = st.number_input("Potassium", 2.0, 8.0, 4.5)
        hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
        packed_cell_volume = st.number_input("Packed Cell Volume", 20, 60, 40)
        white_blood_cell_count = st.number_input("WBC Count", 2000, 20000, 8000)
        red_blood_cell_count = st.number_input("RBC Count", 2.0, 8.0, 4.8)
        hypertension = st.radio("Hypertension", ["yes", "no"])
        diabetes = st.radio("Diabetes Mellitus", ["yes", "no"])
        cad = st.radio("Coronary Artery Disease", ["yes", "no"])
        appetite = st.radio("Appetite", ["good", "poor"])
        pedal_edema = st.radio("Pedal Edema", ["yes", "no"])
        anemia = st.radio("Anemia", ["yes", "no"])

    if st.button("Predict CKD Risk"):
        input_data = {
            'age': age,
            'blood_pressure': blood_pressure,
            'specific_gravity': specific_gravity,
            'albumin': albumin,
            'sugar': sugar,
            'red_blood_cells': 1 if red_blood_cells == "normal" else 0,
            'pus_cell': 1 if pus_cell == "normal" else 0,
            'pus_cell_clumps': 1 if pus_cell_clumps == "present" else 0,
            'bacteria': 1 if bacteria == "present" else 0,
            'blood_glucose_random': blood_glucose,
            'blood_urea': blood_urea,
            'serum_creatinine': serum_creatinine,
            'sodium': sodium,
            'potassium': potassium,
            'haemoglobin': hemoglobin,
            'packed_cell_volume': packed_cell_volume,
            'white_blood_cell_count': white_blood_cell_count,
            'red_blood_cell_count': red_blood_cell_count,
            'hypertension': 1 if hypertension == "yes" else 0,
            'diabetes_mellitus': 1 if diabetes == "yes" else 0,
            'coronary_artery_disease': 1 if cad == "yes" else 0,
            'appetite': 1 if appetite == "good" else 0,
            'peda_edema': 1 if pedal_edema == "yes" else 0,
            'aanemia': 1 if anemia == "yes" else 0,
        }

        pred, prob = predict_ckd(input_data)
        risk = "High Risk" if pred == 0 else "Low Risk"
        color = "red" if pred == 0 else "green"

        st.markdown(f"### 🧾 Prediction for **{patient_name}**:", unsafe_allow_html=True)
        st.markdown(f"### 🩺 Result: <span style='color:{color}; font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
        st.markdown(f"Probability of CKD: **{prob[0]*100:.2f}%**")

# Authentication Interface
def auth_interface():
    st.markdown("<div class='main-header'>Welcome to CKD Predictor</div>", unsafe_allow_html=True)
    auth_tab = st.radio("Login / Signup", ["Login", "Signup"])

    with st.form(key="auth_form"):
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Submit")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if auth_tab == "Signup":
            success, msg = signup(username, password)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        elif auth_tab == "Login":
            if login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials.")

# Entry Point
if __name__ == '__main__':
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    if not st.session_state["logged_in"]:
        auth_interface()
    else:
        show_predictor()
