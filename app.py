import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("preparation_before_the_exam.csv")

df["Did Practical Exercises"] = df["Practical Exercises"].map({"Yes": 1, "No": 0})

def preprocess(df):
    df = df.copy()

    # Fill missing practical exercise averages with 0 (for students who didn't do them)
    if "Average of latest practical exercises" in df.columns:
        df["Average of latest practical exercises"] = df["Average of latest practical exercises"].fillna(0)

    # Fill any other missing numeric values with 0
    df = df.fillna(0)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        if col != "Practical Exercises":  # already replaced by the binary column
            df[col] = le.fit_transform(df[col].astype(str))

    # Drop target-independent column
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
    print(f"{model_name} R2 Score: {r2_score(y_test, y_pred):.2f}")
    print(f"{model_name} MSE: {mean_squared_error(y_test, y_pred):.2f}")
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Prediction vs Reality")
    plt.show()

# Process the entire dataset
df_processed = preprocess(df)
X_train, X_test, y_train, y_test = prepare_features(df_processed)

# Train and evaluate on all students
train_and_evaluate(X_train, X_test, y_train, y_test, LinearRegression(), "Linear Regression - All")
train_and_evaluate(X_train, X_test, y_train, y_test, RandomForestRegressor(random_state=42), "Random Forest - All")

# Page

st.title("Predict the value of matches")
st.divider()

# Numeric Inputs
age = st.number_input("👤 Type your age", min_value=0)
study_time = st.slider("📚 Study Time (hours/day)", min_value=0, max_value=12, value=4)
sleep_duration = st.slider("😴 Average Sleep Duration (hours)", min_value=0, max_value=12, value=7)
practice_tests = st.slider("📝 Practice Tests per Week", min_value=0, max_value=10, value=4)
average_practical = st.slider("📈 Average of Latest Practical Exercises (%)", min_value=0, max_value=100, value=70)

# Selectboxes and Radio buttons for categorical inputs
food_quality = st.selectbox("🍽️ Food Quality", options=["Bad", "Media", "Good", "Great"], index=2)
theoretical_exercises = st.radio("📖 Did Theoretical Exercises?", options=["Yes", "No"])
spaced_study = st.radio("🧠 Used Spaced Study?", options=["Yes", "No"])
motivation = st.selectbox("🔥 Motivation Level", options=["Low", "Media", "High"], index=2)
study_techniques = st.radio("🧪 Use of Study Techniques?", options=["Yes", "No"])
distractions = st.radio("📱 Use of Distractions?", options=["Yes", "No"])
anxiety = st.selectbox("💢 Anxiety Before the Test", options=["Low", "Media", "High"], index=0)
did_practical = st.radio("🛠️ Did Practical Exercises?", options=["Yes", "No"], index=0)

if st.button("🚀 Predict"):

    new_student = {
    "Age": age,
    "Study Time (h/day)": study_time,
    "Average sleep duration in hours": sleep_duration,
    "Practice Tests per Week": practice_tests,
    "Food Quality": food_quality,
    "Theoretical Exercises": theoretical_exercises,
    "Spaced Study": spaced_study,
    "Motivation": motivation,
    "Use of Study Techniques": study_techniques,
    "Use of Distractions": distractions,
    "Anxiety before the test": anxiety,
    "Average of latest practical exercises": average_practical,
    "Did Practical Exercises": did_practical,
    }

    # Turn it into a DataFrame
    new_df = pd.DataFrame([new_student])

    # Preprocess it
    new_df_processed = preprocess(new_df)

    # Ensure it has same columns and order as training data
    missing_cols = set(X_train.columns) - set(new_df_processed.columns)
    for col in missing_cols:
        new_df_processed[col] = 0  # Fill missing ones with default 0
    new_df_processed = new_df_processed[X_train.columns]

    # Predict using a trained model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_score = model.predict(new_df_processed)[0]
    print(f"🎯 Predicted Performance: {predicted_score:.2f}%")

    st.write(f"🎯 Predicted Performance: {predicted_score:.2f}%")
    st.success("Prediction logic goes here...")

# if age:
#     matches = modelo.predict([[age]])[0][0]
#     st.write(f"The apparence {age:.2f} has theses possibles matches {matches:.2f}")