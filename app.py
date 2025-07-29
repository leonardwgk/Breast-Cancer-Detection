from flask import Flask, render_template, request
import numpy as np
import joblib
import sklearn.datasets

app = Flask(__name__)

# Load model
model = joblib.load("breast_cancer_model.pkl")

# Load dataset (untuk ambil nama fitur asli)
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Fungsi untuk ambil nama fitur
def get_feature_names(model):
    # Kalau model pipeline
    if hasattr(model, 'named_steps'):
        # Ambil step terakhir (misal: 'svc')
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, 'feature_names_in_'):
            return final_step.feature_names_in_
    # Kalau model biasa (non-pipeline)
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    # Default: ambil dari dataset sklearn
    return breast_cancer_dataset.feature_names

feature_names = get_feature_names(model)

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form
    features = [float(request.form[f'feature_{i}']) for i in range(30)]
    features_array = np.array(features).reshape(1, -1)

    # Prediksi
    prediction = model.predict(features_array)[0]
    proba = model.predict_proba(features_array)[0][prediction] * 100 if hasattr(model, 'predict_proba') else None
    result = "Benign (Jinak)" if prediction == 1 else "Malignant (Ganas)"

    return render_template('result.html', result=result, proba=proba)

if __name__ == "__main__":
    app.run(debug=True)
