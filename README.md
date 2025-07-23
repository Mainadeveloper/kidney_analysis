# 🩺 Kidney Disease Analysis using Machine Learning

This project focuses on the detection of **Chronic Kidney Disease (CKD)** using a dataset of patient medical records. The goal is to build a machine learning model that can identify individuals likely to have CKD and aid in early diagnosis for timely treatment.

## 📌 Problem Statement
Chronic Kidney Disease is a silent but serious health condition that affects millions globally. Early detection is crucial to avoid kidney failure and costly treatments. This analysis uses data-driven approaches to identify key indicators of CKD and classify patients accordingly.

## 📊 Dataset
The dataset contains both **numerical and categorical** medical features such as:
- Red blood cell count
- Serum creatinine
- Blood pressure
- Albumin levels
- Presence of pus cells
- Hemoglobin levels
- And more...

**Target variable:**
- `class`: 
  - `0` — Patient has Chronic Kidney Disease
  - `1` — Patient does not have CKD

## 🧠 Features of This Project
- Handling missing values using mode and random sampling
- Data visualization using Seaborn and Matplotlib
- Label encoding for categorical variables
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Machine Learning model (e.g., Logistic Regression / Random Forest)
- Evaluation using accuracy, confusion matrix, and classification report

## 🔧 Tools & Libraries
- Python (Pandas, NumPy)
- Seaborn & Matplotlib (for visualization)
- Scikit-learn (for ML models and evaluation)
- Jupyter Notebook

## 📈 Key Insights
- Patients with CKD tend to have lower red blood cell and hemoglobin levels.
- Categorical features like pus cells and albumin presence are significant indicators of CKD.
- Feature engineering and proper data cleaning significantly improve model accuracy.

## ✅ Results
The trained model shows promising accuracy in identifying CKD cases and could potentially support healthcare professionals in decision-making processes.
