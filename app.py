import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("mental_health_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


st.set_page_config(page_title="Mental Health Risk Predictor", layout="centered")
st.title(" Mental Health Risk Predictor")

# Description/About Section
st.markdown("""
### About the Mental Health Risk Prediction Tool

Hi,
I'm **Ifeoluwa Ilesanmi**, am a Computer Science student at **Federal University Oye-Ekiti (FUOYE)**.  
I created this platform as a project to support students in identifying early signs of mental health challenges using machine learning.

This tool analyzes key aspects of a student's life like academic workload, sleep habits, and extracurricular involvement, to help you understand your mental health risk level.

**Mental health matters.** It affects how you think, feel, and function daily. By being proactive and self-aware, you can take meaningful steps to stay emotionally balanced, seek help when needed, and truly thrive in school and beyond.

Let’s normalize mental health conversations starting here, with one simple check.
""")



st.markdown("""
    <style>
    /* Gradient background */
    html, body, .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a9d8f, #6a0572);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Headings */
    h1, h2, h3 {
        color: white;
    }

    /* Form card */
    .stForm {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);
        margin-top: 1rem;
    }

    /* Input and Select boxes - Soft border and padding */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stSlider>div,
    .stRadio>div {
        background-color: rgba(255, 255, 255, 0.12);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);  /* softer border */
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);  /* optional soft shadow */
    }

    /* Improve spacing between widgets */
    .block-container {
        padding: 2rem 2rem;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 2rem 4rem;
    }

    button[kind="primary"]:hover {
        background-color: rgba(255, 255, 255, 0.25);
    }

    /* Caption and smaller text */
    .stCaption, .stMarkdown, .stText {
        color: white !important;
    }

    /* Hide Streamlit header and footer */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    

    </style>
""", unsafe_allow_html=True)




# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=10, max_value=50, value=20)
        academic_year = st.radio("Current Academic Year", ["100", "200", "300", "400", "500"])
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=5.0, step=0.1)
        residential = st.radio("Residential Status", ["On-campus", "Off-campus"])
        sports = st.slider("Sports Engagement", 0, 5, 2)

    with col2:
        sleep = st.number_input("Average Sleep Time (hours/day)", min_value=0.0, max_value=24.0, step=0.5)
        extracurricular = st.slider("Extracurricular Involvement", 0, 5, 2)
        workload = st.slider("Academic Workload", 0, 5, 3)
        diet = st.slider("Diet Quality", 0, 5, 2)
        finance = st.slider("Financial Pressure", 0, 5, 3)
        social = st.slider("Social Relationships", 0, 5, 2)
        anxiety = st.slider("Anxiety Level", 0, 5, 3)
        academic_pressure = st.slider("Academic Pressure", 0, 5, 3)

    submitted = st.form_submit_button("Check Risk")

# Prediction
if submitted:
    input_dict = {
        "Gender": gender,
        "Age": age,
        "Current_academic_year": academic_year,
        "CGPA": cgpa,
        "Residential_status": residential,
        "Sports_engagement": sports,
        "Average_sleeptime": sleep,
        "ExtraCurricular_Involvement": extracurricular,
        "Academic_workload": workload,
        "Diet_Quality": diet,
        "Financial_pressure": finance,
        "Social_relationships": social,
        "Anxiety": anxiety,
        "Academic_pressure": academic_pressure
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    # Ensure all columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]  # reorder

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Mental Health Risk")
    else:
        st.success("✅ Low Mental Health Risk")

st.markdown("""
    <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.3); margin-top: 3rem;"/>

    <div style='text-align: center; padding-top: 10px; color: white; font-size: 14px;'>
        <em>Created with 💚 by Ifeoluwa Ilesanmi</em>
    </div>
""", unsafe_allow_html=True)
