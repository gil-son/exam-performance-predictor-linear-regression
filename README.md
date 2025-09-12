  <div align="center">
    <img src="https://cdn-icons-png.flaticon.com/512/1549/1549879.png" width="10%">
  </div>

# Predict Performance Before the Exam (Streamlit App)

This is a simple, interactive web app built using **Streamlit** and **Scikit-learn** that predicts a student's likely performance before an exam based on behavioral and preparation factors.

But let’s take a moment to reflect on this project...

Often, we follow a learning journey focused on mastering specific topics — and eventually take an exam to prove our skills.

But it’s no secret that within what’s effective, there’s always something more efficient.

A car can get you from one state to another, just like a plane can — but when it comes to efficiency, the plane wins.

That’s exactly the idea behind this project: making the exam prep journey more efficient.
How? By analyzing your study habits, sleep quality, routines, motivation, and more!

This tool uses a machine learning model I trained — still in its early (and far from final) version 😅 — but it already provides useful insights for self-assessment.

Key points:
- You don’t need this model — it’s just an aid.
- The result is a projection, not a definitive answer.
- It helps identify areas to improve.
- It's great for self-reflection.
- It’s open source — contributions welcome!

---

## 📁 Files Needed

Place all the following files in the same folder:

```
├── app.py
├── human_verification.py
├── trained_performance_model.joblib
├── requirements.txt
```

---

## ⚙️ How to Run the App Locally

### 1. Clone or Copy the Files

If not using Git, just upload/copy the files into a project folder.

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will be accessible at: [http://localhost:8501](http://localhost:8501)

<div align="center">
<img src="https://cdn-icons-png.flaticon.com/512/18310/18310927.png" width="10%"/> 
</div>

## Predict Your Score Online

  1. Instructions and **Access the Link**
   - This is the **Beta version**
   - Available daily from 10 AM to 7 PM (UTC-3 / Brasília Time)
   - You can test it **once per day**
   - **Live App**: [Try it here](http://35.175.151.115:8501/)

  2. **Solve a challenge to prove you are a human**  
     Solve the clock chanllenge to get form access.

  <div align="center">
    <img src="https://i.ibb.co/PshyBqTR/p01.png" width="70%">
  </div>

  3. **Provide Your Information**  
     Enter your basic personal and study-related details, then click the **Predict Performance** button. A Linear Regression model will analyze your inputs to estimate your exam score.

  <div align="center">
    <img src="https://i.ibb.co/qMGJmGwj/p03.png" width="70%">
  </div>

  4. **Get an Approximate Score**  
     You'll receive a predicted performance percentage.

  <div align="center">
    <img src="https://i.ibb.co/kgxZFqXC/p04.png" width="70%">
  </div>
