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
    .stNumberInput>div>div>input,
    .stTextInput > div > div > input,
    .stTextInput > div > div > textarea,
    .stSelectbox>div>div,
    .stSlider>div,
    .stRadio>div {
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);  /* softer border */
        border-radius: 10px;
        padding: 0.6rem 1rem;
        
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
        name = st.text_input("Enter your name")
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=10, max_value=50)
        academic_year = st.radio("Current Academic Year", ["100", "200", "300", "400", "500"])
        cgpa = st.number_input("CGPA", min_value=1.0, max_value=5.0, step=0.1)
        residential = st.radio("Residential Status", ["On-campus", "Off-campus"])
        sports = st.slider("Sports Engagement", 0, 5)

    with col2:
        sleep = st.number_input("Average Sleep Time (hours/day)", min_value=1.0, max_value=24.0, step=0.5)
        extracurricular = st.slider("Extracurricular Involvement", 0, 5)
        workload = st.slider("Academic Workload", 0, 5)
        diet = st.slider("Diet Quality", 0, 5)
        finance = st.slider("Financial Pressure", 0, 5)
        social = st.slider("Social Relationships", 0, 5)
        anxiety = st.slider("Anxiety Level", 0, 5)
        academic_pressure = st.slider("Academic Pressure", 0, 5)

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

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]  # prediction is an integer

    if prediction == 1:
        st.error(
            f"⚠️ {name}, your result shows a **High Mental Health Risk**.\n\n"
        )
        st.write(
            """
            **Hey there, thank you for taking this mental health check.**  
            Your result indicates a *high risk*. This doesn't mean something is wrong with you, it simply signals that you may be under more stress than usual, and it’s okay to need support.

            > 💡 **Here’s what you can do:**
            - Talk to someone you trust, a friend, mentor, or school counselor.
            - Prioritize sleep, food, and rest. It’s not laziness, it’s self-care.
            - Reduce your academic pressure where possible. Don’t hesitate to ask for help.
            - You’re not alone. Many students feel this way and things *can* get better.

            **Your mental health matters.** And the fact that you're checking in? That’s courage. Take it one step at a time.

            > 💬 *If you’d like help finding support, reach out to school health services or someone close to you.*
            """
        )
    else:
        st.success(
            f"✅ {name}, your result shows a **Low Mental Health Risk**.\n\n"
        )
        st.write(
            """
            **Great job taking time for your mental health!**  
            Your result indicates a *low risk* and that’s awesome! You seem to be handling things well right now.

            > 💡 **Keep it up by:**
            - Staying connected to friends and activities you enjoy.
            - Taking breaks to recharge, even when things feel okay.
            - Checking in on your friends too. A kind word can go a long way.

            Your well-being is important. Stay mindful, and keep being kind to yourself.
            """
        )


st.markdown("""
    <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.3); margin-top: 3rem;"/>

    <div style='text-align: center; padding-top: 10px; color: white; font-size: 14px;'>
        <em>Created with 💚 by Ifeoluwa Ilesanmi</em>
    </div>
""", unsafe_allow_html=True)

