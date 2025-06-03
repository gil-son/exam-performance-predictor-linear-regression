import pandas as pd
import random
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from human_verification import run_human_verification

# --- Constants ---
PREDICTION_BLOCK_FILE = "prediction_block_until.txt"
PREDICTION_COOLDOWN_HOURS = 24  # Cooldown set to 1 day

# --- Load the pre-trained model ---
try:
    loaded_model = joblib.load("trained_performance_model.joblib")
    st.info("✅ Pre-trained model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Pre-trained model file not found. Please train and save the model first.")
    loaded_model = None

# --- Load and preprocess training data ---
df = pd.read_csv("preparation_before_the_exam.csv")
df["Did Practical Exercises"] = df["Practical Exercises"].map({"Yes": 1, "No": 0})

def preprocess(df):
    df = df.copy()
    if "Average of latest practical exercises" in df.columns:
        df["Average of latest practical exercises"] = df["Average of latest practical exercises"].fillna(0)
    df = df.fillna(0)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        if col != "Practical Exercises":
            df[col] = le.fit_transform(df[col].astype(str))
    df = df.drop(columns=["Practical Exercises"], errors="ignore")
    return df

def predict_performance(model, user_input_df, training_columns):
    user_input_processed = preprocess(user_input_df)
    missing_cols = set(training_columns) - set(user_input_processed.columns)
    for col in missing_cols:
        user_input_processed[col] = 0
    user_input_processed = user_input_processed[training_columns]
    prediction = model.predict(user_input_processed)[0]
    return prediction

# --- Prediction Block Helpers ---
def _load_prediction_block_until():
    if os.path.exists(PREDICTION_BLOCK_FILE):
        ts = open(PREDICTION_BLOCK_FILE).read().strip()
        return datetime.fromisoformat(ts)
    return None

def _save_prediction_block_until(dt: datetime):
    with open(PREDICTION_BLOCK_FILE, "w") as f:
        f.write(dt.isoformat())

# --- Style tweaks ---
st.markdown("""
    <style>
        p, label { font-size: 24px !important; }
        div[data-baseweb="select"] div { font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

# --- Human Verification ---
if "human_verified" not in st.session_state:
    st.session_state.human_verified = False

if not st.session_state.human_verified:
    run_human_verification()
    st.stop()

st.success("🎉 You're verified as human!")

# --- Prediction Cooldown Check ---
now = datetime.now()
block_until = _load_prediction_block_until()

if block_until and now < block_until:
    remaining_hours = int((block_until - now).total_seconds() // 3600)
    st.warning(f"⏳ You’ve recently predicted. Please wait **{remaining_hours} hours** before trying again.")

    st.stop()

# --- Main App ---
st.title("🎯 Predict the Performance Before the Exam")
st.divider()

with st.form("input_form"):
    age = st.number_input("👤 Age", min_value=0, value=18)
    study_time = st.slider("📚 Study Time (hours/day)", 0, 12, 4)
    sleep = st.slider("😴 Sleep Duration (hours)", 0, 12, 7)
    tests_per_week = st.slider("📝 Practice Tests per Week", 0, 10, 4)
    avg_exercises = st.slider("📈 Avg Practical Exercises (%)", 0, 100, 70)
    food_quality = st.selectbox("🍽️ Food Quality", ["Bad", "Media", "Good", "Great"])
    did_practical = st.radio("🧪 Did Practical Exercises?", ["Yes", "No"])
    did_theoretical = st.radio("📖 Did Theoretical Exercises?", ["Yes", "No"])
    spaced_study = st.radio("🧠 Used Spaced Study?", ["Yes", "No"])
    motivation = st.selectbox("🔥 Motivation Level", ["Low", "Media", "High"])
    used_techniques = st.radio("🧪 Used Study Techniques?", ["Yes", "No"])
    used_distractions = st.radio("📱 Used Distractions?", ["Yes", "No"])
    anxiety = st.selectbox("💢 Anxiety Before the Test", ["Low", "Media", "High"])
    
    submitted = st.form_submit_button("🚀 Predict Performance")

if submitted and loaded_model:
    new_student = {
        "Age": age,
        "Study Time (h/day)": study_time,
        "Average sleep duration in hours": sleep,
        "Practice Tests per Week": tests_per_week,
        "Average of latest practical exercises": avg_exercises,
        "Food Quality": food_quality,
        "Practical Exercises": did_practical,
        "Theoretical Exercises": did_theoretical,
        "Spaced Study": spaced_study,
        "Motivation": motivation,
        "Use of Study Techniques": used_techniques,
        "Use of Distractions": used_distractions,
        "Anxiety before the test": anxiety
    }
    
    new_df = pd.DataFrame([new_student])
    new_df["Did Practical Exercises"] = new_df["Practical Exercises"].map({"Yes": 1, "No": 0})
    processed_new_df = preprocess(new_df)
    training_columns = [col for col in df.columns if col != "Performance (%)" and col != "Practical Exercises"]

    try:
        prediction = predict_performance(loaded_model, processed_new_df, training_columns)
        capped_prediction = min(prediction, 100)

        if prediction > 100:
            prediction = 100

        st.success(f"🎯 Predicted Performance: `{prediction:.2f}%`")

        # --- Conditional Advice ---
        if prediction < 70:
            st.warning("⚠️ **Attention!** Your predicted score is below 70%. Consider reviewing your study habits and preparation strategies.")
        elif prediction < 72:
            st.info("📘 **Almost there!** You're close to a good score. A few improvements could make a big difference!")
        elif prediction >= 73:
            st.success("🎉 **Great job!** Your preparation looks strong. Keep it up and trust your performance!")

        # Save cooldown time
        next_allowed = datetime.now() + timedelta(hours=PREDICTION_COOLDOWN_HOURS)
        _save_prediction_block_until(next_allowed)

        # --- Donut Chart ---
        fig1, ax1 = plt.subplots()
        wedges, texts = ax1.pie(
            [capped_prediction, 100 - capped_prediction],
            labels=['Possible Score', 'Remaining'],
            colors=['#00c49a', '#e0e0e0'],
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3}
        )

        ax1.set(aspect="equal")
        ax1.text(0, 0, f"{prediction:.1f}%", ha='center', va='center', fontsize=24, fontweight='bold', color='#00c49a')
        st.markdown("### 🟢 Performance Overview")
        st.pyplot(fig1)

        # --- Simulated Weekly Performance ---
        weeks_count = max(1, tests_per_week)
        weeks = np.arange(1, weeks_count + 1)
        simulated_scores = np.clip(prediction + np.random.normal(1, 1, size=weeks_count).cumsum(), 0, 100)

        fig2, ax2 = plt.subplots()
        ax2.plot(weeks, simulated_scores, marker='o', color='#1f77b4', linewidth=2)
        for i, score in enumerate(simulated_scores):
            ax2.text(weeks[i], score + 1, f"{score:.1f}%", ha='center', fontsize=10)

        ax2.set_title("📈 Simulated Weekly Performance")
        ax2.set_xlabel("Week")
        ax2.set_ylabel("Performance (%)")
        ax2.set_ylim(0, 110)
        ax2.grid(True)

        st.pyplot(fig2)
        st.markdown("### 📈 Simulated Weekly Performance")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")