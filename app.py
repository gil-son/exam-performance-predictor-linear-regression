import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load and preprocess the dataset
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

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"### {model_name} Evaluation")
    st.write(f"- R2 Score: `{r2_score(y_test, y_pred):.2f}`")
    st.write(f"- MSE: `{mean_squared_error(y_test, y_pred):.2f}`")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} Prediction vs Reality")
    st.pyplot(fig)

# Streamlit UI
st.title("🎯 Predict the Performance Before the Exam")
st.divider()

# Session state for steps
if "step" not in st.session_state:
    st.session_state.step = 0
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "confirmed_last_input" not in st.session_state:
    st.session_state.confirmed_last_input = False

# Input steps (fields)
steps = [
    ("Age", lambda: st.number_input("👤 Type your age", min_value=0, value=st.session_state.inputs.get("Age", 18), key="Age")),
    ("Study Time (h/day)", lambda: st.slider("📚 Study Time (hours/day)", 0, 12, value=st.session_state.inputs.get("Study Time (h/day)", 4), key="Study Time (h/day)")),
    ("Average sleep duration in hours", lambda: st.slider("😴 Average Sleep Duration (hours)", 0, 12, value=st.session_state.inputs.get("Average sleep duration in hours", 7), key="Average sleep duration in hours")),
    ("Practice Tests per Week", lambda: st.slider("📝 Practice Tests per Week", 0, 10, value=st.session_state.inputs.get("Practice Tests per Week", 4), key="Practice Tests per Week")),
    ("Average of latest practical exercises", lambda: st.slider("📈 Average of Latest Practical Exercises (%)", 0, 100, value=st.session_state.inputs.get("Average of latest practical exercises", 70), key="Average of latest practical exercises")),
    ("Food Quality", lambda: st.selectbox("🍽️ Food Quality", ["Bad", "Media", "Good", "Great"], index=["Bad", "Media", "Good", "Great"].index(st.session_state.inputs.get("Food Quality", "Good")), key="Food Quality")),
    ("Theoretical Exercises", lambda: st.radio("📖 Did Theoretical Exercises?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Theoretical Exercises", "Yes")), key="Theoretical Exercises")),
    ("Spaced Study", lambda: st.radio("🧠 Used Spaced Study?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Spaced Study", "Yes")), key="Spaced Study")),
    ("Motivation", lambda: st.selectbox("🔥 Motivation Level", ["Low", "Media", "High"], index=["Low", "Media", "High"].index(st.session_state.inputs.get("Motivation", "High")), key="Motivation")),
    ("Use of Study Techniques", lambda: st.radio("🧪 Use of Study Techniques?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Use of Study Techniques", "Yes")), key="Use of Study Techniques")),
    ("Use of Distractions", lambda: st.radio("📱 Use of Distractions?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs.get("Use of Distractions", "Yes")), key="Use of Distractions")),
    ("Anxiety before the test", lambda: st.selectbox("💢 Anxiety Before the Test", ["Low", "Media", "High"], index=["Low", "Media", "High"].index(st.session_state.inputs.get("Anxiety before the test", "Low")), key="Anxiety before the test")),
]

# Navigation buttons
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

# Show current input step
if st.session_state.step < len(steps):
    current_step_label, current_input = steps[st.session_state.step]
    st.session_state.inputs[current_step_label] = current_input()

# Predict Button (only after confirmation)
if st.session_state.step == len(steps) - 1 and st.session_state.confirmed_last_input:
    st.markdown("---")
    st.success("✅ All inputs completed!")

    if st.button("🚀 Predict & Train"):

        # Create DataFrame for the new student from collected inputs
        new_student = {key: st.session_state.inputs[key] for key, _ in steps}
        new_df = pd.DataFrame([new_student])

        # Preprocess the original dataset
        processed_df = preprocess(df)
        X_train, X_test, y_train, y_test = prepare_features(processed_df)

        # Preprocess the new input
        new_df_processed = preprocess(new_df)

        # Align the new input columns with training columns
        missing_cols = set(X_train.columns) - set(new_df_processed.columns)
        for col in missing_cols:
            new_df_processed[col] = 0
        new_df_processed = new_df_processed[X_train.columns]

        # Train the model and make prediction
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_score = model.predict(new_df_processed)[0]

        # Show prediction result
        st.success(f"🎯 Predicted Performance: `{predicted_score:.2f}%`")

        # Optionally retrain the model with the new input
        st.markdown("📊 Retraining model with the new input data...")
        df_combined = pd.concat([df, new_df], ignore_index=True)
        df_combined_processed = preprocess(df_combined)
        X_train, X_test, y_train, y_test = prepare_features(df_combined_processed)
        model.fit(X_train, y_train)

        # Evaluate and display updated model metrics
        train_and_evaluate(X_train, X_test, y_train, y_test, model, "Linear Regression - Updated")