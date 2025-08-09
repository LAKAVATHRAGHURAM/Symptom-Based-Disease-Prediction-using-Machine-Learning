# 🩺 Symptom-Based Disease Prediction using Machine Learning

## 📌 Overview
This project predicts **possible diseases** based on **user-input symptoms** using a trained Machine Learning model.  
It’s designed for **quick health assessments** and can be integrated into **chatbots** or **healthcare assistant applications**.  
> ⚠ Disclaimer: This tool is for **educational purposes only** and should not replace professional medical advice.

---

## 🚀 Features
- 🧾 **Multiple Symptom Input** – Enter more than one symptom for better accuracy.
- 🎯 **Top Disease Predictions** – Shows most probable diseases with accuracy scores.
- 🤖 **Machine Learning Model** – Uses **Random Forest Classifier** for high accuracy.
- 🌐 **Web Interface (Flask)** – Simple and interactive user interface.
- 📂 **Extendable** – Can be updated with more symptoms/diseases for better coverage.

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:** Pandas, NumPy, Scikit-learn, Flask  
- **Model Serialization:** Pickle  
- **Interface:** CLI & Web (Flask)

---

## 📊 Dataset
- The dataset contains **symptoms** as features and **diseases** as labels.
- Preprocessing steps:
  - Handle missing values
  - Convert symptoms to numerical format
  - One-hot encoding for categorical features

---

## 🧠 Model Details
- **Algorithm:** Random Forest Classifier
- **Training Accuracy:** ~98%
- **Evaluation:** Tested with real and synthetic symptom combinations
- **File:** `disease_model.pkl` (serialized trained model)

---

