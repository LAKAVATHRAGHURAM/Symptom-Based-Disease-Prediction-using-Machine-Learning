# app.py
import streamlit as st
import joblib
import numpy as np
from trie import Trie
import pandas as pd

# Load model and symptoms list
model = joblib.load("model.pkl")
symptoms = joblib.load("symptoms_list.pkl")

# Build Trie for autocomplete
trie = Trie()
for s in symptoms:
    trie.insert(s)

# Load disease precautions
precaution_df = pd.read_csv("data/symptom_precaution.csv")
precautions = {}
for _, row in precaution_df.iterrows():
    precautions[row['Disease']] = [
        row['Precaution_1'], row['Precaution_2'], row['Precaution_3']
    ]

# Load disease descriptions
desc_df = pd.read_csv("data/symptom_Description.csv")
symptom_desc = dict(zip(desc_df['Disease'], desc_df['Description']))

# Streamlit UI
st.title("🩺 Symptom to Disease Predictor")

st.write("Start typing a symptom below, then pick from suggestions:")

# Autocomplete
prefix = st.text_input("Symptom search:")
suggestions = trie.get_suggestions(prefix)
selected_symptoms = st.multiselect("Select your symptoms:", suggestions)

if selected_symptoms:
    st.write("### Selected Symptoms & Their Descriptions:")
    for s in selected_symptoms:
        desc = symptom_desc.get(s, "No description available.")
        st.markdown(f"- **{s.capitalize()}**: {desc}")

if st.button("🔎 Predict Disease"):
    if selected_symptoms:
        input_vec = [1 if s in selected_symptoms else 0 for s in symptoms]
        prediction = model.predict([input_vec])[0]

        st.success(f"✅ **Predicted Disease:** {prediction}")

        # Show precautions if available
        precaution_list = precautions.get(prediction, [])
        if precaution_list:
            st.write("### 🛡️ Recommended Precautions:")
            for p in precaution_list:
                st.write(f"- {p}")
    else:
        st.warning("Please select at least one symptom to predict a disease.")
