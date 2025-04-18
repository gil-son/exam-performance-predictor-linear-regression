import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Load and preprocess your full training dataset ---
df_train = pd.read_csv("preparation_before_the_exam.csv")
df_train["Did Practical Exercises"] = df_train["Practical Exercises"].map({"Yes": 1, "No": 0})

def preprocess_train(df):
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

processed_df_train = preprocess_train(df_train)
X = processed_df_train.drop(columns=["Performance (%)"])
y = processed_df_train["Performance (%)"]

# --- Train the model ---
model = LinearRegression()
model.fit(X, y)

# --- Save the trained model using joblib ---
filename = "trained_performance_model.joblib"
joblib.dump(model, filename)

print(f"✅ Trained model saved to: {filename}")