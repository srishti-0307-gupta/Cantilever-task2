# Cantilever-task2
This project is a machine learning-based web application designed to detect fraudulent credit card transactions. It leverages data preprocessing, feature encoding, and classification algorithms to analyze transaction data and predict whether a transaction is legitimate or fraudulent.
# 💳 Credit Card Fraud Detection System

A machine learning-powered web application to detect fraudulent credit card transactions. Built with a Flask backend and a simple HTML frontend, this project allows users to input transaction data and get real-time predictions on whether the transaction is genuine or fraudulent.

---



## 🧠 Tech Stack

| Category     | Technologies                      |
|--------------|------------------------------------|
| Language     | Python 3, HTML, CSS                |
| Framework    | Flask                              |
| Libraries    | Pandas, NumPy, scikit-learn, pickle|
| Frontend     | HTML5, CSS3                        |
| Model        | Logistic Regression / Random Forest / any classifier you used |

---

## 📁 Folder Structure

credit-card-fraud-detection/
│
├── templates/
│ └── index.html 
│ └── result.html
├── fraud_model.pkl # Trained ML model + encoders (serialized)
├── app.py # Flask backend for routing & prediction
├── train_model.py 
