  <div align="center">
    <img src="https://cdn-icons-png.flaticon.com/512/1549/1549879.png" width="10%">
  </div>

# Predict Performance Before the Exam (Streamlit App)

This is a simple, interactive web app built using **Streamlit** and **Scikit-learn** that predicts a student's likely performance before an exam based on behavioral and preparation factors.

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
   - **Live App**: [Try it here](http://54.234.171.149:8501/)

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

<hr/>

  <div align="center">
    <img src="https://i.ibb.co/kgNSnpv/git-support.png">
  </div>
