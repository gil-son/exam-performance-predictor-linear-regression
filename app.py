import pandas as pd
import random
import streamlit as st
import joblib  # For saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from human_verification import run_human_verification

# --- Load the pre-trained model ---
try:
    loaded_model = joblib.load("trained_performance_model.joblib")
    st.info("✅ Pre-trained model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Pre-trained model file not found. Please train and save the model first.")
    loaded_model = None

# --- Load and Preprocess Training Data (for preprocessing new input) ---
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

def prepare_features(df):
    if len(df) < 2:
        raise ValueError(f"Not enough data to split. Only {len(df)} sample(s) available.")
    X = df.drop(columns=["Performance (%)"])
    y = df["Performance (%)"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def predict_performance(model, user_input_df, training_columns):
    user_input_processed = preprocess(user_input_df)
    missing_cols = set(training_columns) - set(user_input_processed.columns)
    for col in missing_cols:
        user_input_processed[col] = 0
    user_input_processed = user_input_processed[training_columns]
    prediction = model.predict(user_input_processed)[0]
    return prediction

# --- Session State Initialization ---
if "human_check_number" not in st.session_state:
    st.session_state.human_check_number = random.randint(1, 100)
if "human_verified" not in st.session_state:
    st.session_state.human_verified = False
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "confirmed_last_input" not in st.session_state:
    st.session_state.confirmed_last_input = False

# --- Human Verification ---
run_human_verification()

# --- Main App Content ---
st.success("🎉 You're verified as human!")
st.write("You can now access the rest of the app.")

# ------------------ 🎯 Main UI ------------------
st.title("🎯 Predict the Performance Before the Exam")
st.divider()

# --- Input Steps ---
steps = [
    ("Age", lambda: st.number_input("👤 Type your age", min_value=0, value=st.session_state.inputs.get("Age", 18), key="Age")),
    ("Study Time (h/day)", lambda: st.slider("📚 Study Time (hours/day)", 0, 12, value=st.session_state.inputs.get("Study Time (h/day)", 4), key="Study Time (h/day)")),
    ("Average sleep duration in hours", lambda: st.slider("😴 Sleep Duration (hours)", 0, 12, value=st.session_state.inputs.get("Average sleep duration in hours", 7), key="Average sleep duration in hours")),
    ("Practice Tests per Week", lambda: st.slider("📝 Practice Tests per Week", 0, 10, value=st.session_state.inputs.get("Practice Tests per Week", 4), key="Practice Tests per Week")),
    ("Average of latest practical exercises", lambda: st.slider("📈 Avg Practical Exercises (%)", 0, 100, value=st.session_state.inputs.get("Average of latest practical exercises", 70), key="Average of latest practical exercises")),
    ("Food Quality", lambda: st.selectbox("🍽️ Food Quality", ["Bad", "Media", "Good", "Great"], index=["Bad", "Media", "Good", "Great"].index(st.session_state.inputs.get("Food Quality", "Good")), key="Food Quality")),
    ("Practical Exercises", lambda: st.radio("🧪 Did Practical Exercises?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Practical Exercises", "No")), key="Practical Exercises")),
    ("Theoretical Exercises", lambda: st.radio("📖 Did Theoretical Exercises?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Theoretical Exercises", "Yes")), key="Theoretical Exercises")),
    ("Spaced Study", lambda: st.radio("🧠 Used Spaced Study?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("Spaced Study", "Yes")), key="Spaced Study")),
    ("Motivation", lambda: st.selectbox("🔥 Motivation Level", ["Low", "Media", "High"], index=["Low", "Media", "High"].index(st.session_state.get("Motivation", "High")), key="Motivation")),
    ("Use of Study Techniques", lambda: st.radio("🧪 Used Study Techniques?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("Use of Study Techniques", "Yes")), key="Use of Study Techniques")),
    ("Use of Distractions", lambda: st.radio("📱 Used Distractions?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("Use of Distractions", "Yes")), key="Use of Distractions")),
    ("Anxiety before the test", lambda: st.selectbox("💢 Anxiety Before the Test", ["Low", "Media", "High"], index=["Low", "Media", "High"].index(st.session_state.get("Anxiety before the test", "Low")), key="Anxiety before the test")),
]

# --- Navigation and Input Display ---
if st.session_state.step < len(steps):
    current_step_label, current_input = steps[st.session_state.step]
    st.subheader(f"Step {st.session_state.step + 1}: {current_step_label}")
    st.session_state.inputs[current_step_label] = current_input()

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.step > 0 and st.button("⬅️ Previous"):
            st.session_state.step -= 1
            st.session_state.confirmed_last_input = False
    with col2:
        if st.session_state.step < len(steps) - 1:
            if st.button("➡️ Next"):
                st.session_state.step += 1
        elif st.session_state.step == len(steps) - 1:
            if st.button("✅ Confirm Last Input"):
                st.session_state.confirmed_last_input = True

# --- Final Prediction ---
if st.session_state.step == len(steps) - 1 and st.session_state.confirmed_last_input:
    st.markdown("---")
    st.success("✅ All inputs completed!")

    if st.button("🚀 Predict Performance"):
        if loaded_model:
            new_student = {key: st.session_state.inputs[key] for key, _ in steps}
            new_df = pd.DataFrame([new_student])

            # Check if 'Practical Exercises' was part of the user inputs
            if "Practical Exercises" in new_df.columns:
                new_df["Did Practical Exercises"] = new_df["Practical Exercises"].map({"Yes": 1, "No": 0})
            else:
                new_df["Did Practical Exercises"] = 0
                st.warning("⚠️ 'Practical Exercises' information not provided. Assuming 'No'.")

            # Preprocess the new input
            processed_new_df = preprocess(new_df)

            # Get the columns the model was trained on
            training_columns = [col for col in df.columns if col != "Performance (%)" and col != "Practical Exercises"] # Adjust based on your actual training columns

            # Make the prediction
            predicted_score = predict_performance(loaded_model, processed_new_df, training_columns)
            st.success(f"🎯 Predicted Performance: `{predicted_score:.2f}%`")
        else:
            st.error("⚠️ Pre-trained model not loaded. Cannot make prediction.")