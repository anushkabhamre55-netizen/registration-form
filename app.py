import streamlit as st
import pandas as pd
import joblib
import os
from src.auth import verify_user

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "job_pipeline.joblib")

# -------------------- App Configuration --------------------
st.set_page_config(
    page_title="Job Eligibility Predictor",
    layout="centered",
    page_icon="🧭",
)

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
    body {
        background-color: #F4F6F8;
        font-family: 'Helvetica Neue', sans-serif;
        color: #1C1C1C;
    }

    h1, h2, h3 {
        text-align: center;
        color: #0B3D91;
        font-weight: 700;
    }

    /* Login form card styling */
    div[data-testid="stForm"][aria-label="login_form"] {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 2rem 2.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        width: 350px;
        margin: 2rem auto;
        text-align: center;
    }

    /* Login input fields */
    input, select, textarea {
        border-radius: 8px !important;
        border: 1px solid #C5C5C5 !important;
        padding: 0.6rem !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        text-align: center;
    }

    input:focus, select:focus, textarea:focus {
        border-color: #0B3D91 !important;
        box-shadow: 0 0 0 2px rgba(11,61,145,0.2) !important;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0B3D91, #1F5BB5);
        color: white !important;
        font-weight: 600;
        padding: 0.8rem;
        border-radius: 10px;
        font-size: 1rem;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1F5BB5, #3A7FD5);
        transform: scale(1.03);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #E8F0FA;
    }

    /* Messages */
    .stSuccess {
        background-color: #DFF2E1 !important;
        border-left: 5px solid #2E7D32 !important;
    }
    .stError {
        background-color: #FDECEA !important;
        border-left: 5px solid #C62828 !important;
    }

    /* Center login title & description */
    .stMarkdown h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown p {
        font-size: 1rem;
        color: #4D4D4D;
    }

    /* -------- Table Styling (Prediction Output) -------- */
    [data-testid="stDataFrame"] table {
        border-collapse: collapse !important;
        width: 100% !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        margin-top: 1rem !important;
    }

    [data-testid="stDataFrame"] thead tr th {
        background-color: #0B3D91 !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: center !important;
        border: 1px solid #E0E0E0 !important;
        padding: 10px !important;
    }

    [data-testid="stDataFrame"] tbody tr td {
        text-align: center !important;
        border: 1px solid #E0E0E0 !important;
        padding: 10px !important;
    }

    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #F7F9FC !important;
    }

    [data-testid="stDataFrame"] {
        max-height: 350px !important;
        overflow-y: auto !important;
    }

    </style>
""", unsafe_allow_html=True)

# --------------------------- STATE ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = None

# --------------------------- LOGIN ---------------------------
def show_login():
    st.markdown("<h1>Login to Access the Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Please enter your credentials to continue.</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        st.text_input("Email", key="login_email")
        st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")
        if submitted:
            email = st.session_state.login_email.strip()
            password = st.session_state.login_password
            if verify_user(email, password):
                st.session_state.logged_in = True
                st.session_state.email = email
                st.success("Logged in successfully! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid email or password. Please try again.")

# --------------------------- PREDICTOR ---------------------------
def show_predictor():
    st.sidebar.header("User Info")
    st.sidebar.success(f"Logged in as: {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = None
        st.rerun()

    st.markdown("<h1>Job Eligibility Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Fill in your details below to check which job role fits you best.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=25)
            education = st.selectbox("Education", ["High School", "Diploma", "Bachelors", "Masters", "PhD"])
            years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
            skills_count = st.number_input("Relevant Skills Count", min_value=0, max_value=20, value=3)
            certifications = st.number_input("Certifications Count", min_value=0, max_value=20, value=0)

        with col2:
            communication = st.slider("Communication (0–10)", 0, 10, 6)
            leadership = st.slider("Leadership (0–10)", 0, 10, 4)
            technical = st.slider("Technical (0–10)", 0, 10, 5)
            willing_to_travel = st.selectbox("Willing to Travel?", ["No", "Yes"])
            preferred_domain = st.selectbox("Preferred Domain", ["IT", "Finance", "Operations", "Marketing", "HR", "Manufacturing"])

        submit = st.form_submit_button("Predict Eligibility")

    if submit:
        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Please train the model first.")
            return
        try:
            pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error("Failed to load the model: " + str(e))
            return

        sample = pd.DataFrame([{
            "age": int(age),
            "education": education,
            "years_experience": int(years_experience),
            "skills_count": int(skills_count),
            "communication": int(communication),
            "leadership": int(leadership),
            "technical": int(technical),
            "certifications": int(certifications),
            "willing_to_travel": 1 if willing_to_travel == "Yes" else 0,
            "preferred_domain": preferred_domain
        }])

        try:
            preds = pipeline.predict(sample)
            st.success(f"Predicted Role: {preds[0]}")
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(sample)
                st.markdown("<h3 style='color:#0B3D91; text-align:center;'>Probability Breakdown</h3>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(proba, columns=pipeline.classes_))
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# --------------------------- MAIN ---------------------------
if not st.session_state.logged_in:
    show_login()
else:
    show_predictor()
