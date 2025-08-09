from flask import Flask, render_template, request
import joblib
import pandas as pd
from trie import Trie

app = Flask(__name__)

# Load model and symptoms list
model = joblib.load("model.pkl")
symptoms = joblib.load("symptoms_list.pkl")

# Build Trie for autocomplete
trie = Trie()
for s in symptoms:
    trie.insert(s)

# Load disease precautions
precaution_df = pd.read_csv("data/symptom_precaution.csv")
precautions = {
    row['Disease']: [row['Precaution_1'], row['Precaution_2'], row['Precaution_3']]
    for _, row in precaution_df.iterrows()
}

# Load disease descriptions
desc_df = pd.read_csv("data/symptom_Description.csv")
disease_desc = dict(zip(desc_df['Disease'], desc_df['Description']))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_symptoms = []
    selected_desc = []
    precaution_list = []

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")
        if selected_symptoms:
            # Vectorize input
            input_vec = [1 if s in selected_symptoms else 0 for s in symptoms]
            prediction = model.predict([input_vec])[0]

            # Get descriptions for selected symptoms
            selected_desc = [(s, disease_desc.get(s, "No description available")) for s in selected_symptoms]

            # Get precautions
            precaution_list = precautions.get(prediction, [])

    return render_template(
        "index.html",
        symptoms=symptoms,
        selected_symptoms=selected_symptoms,
        selected_desc=selected_desc,
        prediction=prediction,
        precautions=precaution_list
    )

if __name__ == "__main__":
    app.run(debug=True)
