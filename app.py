from flask import Flask, render_template, request
import numpy as np
import joblib
import sklearn.datasets

app = Flask(__name__)

model = joblib.load("breast_cancer_model.pkl")

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

def get_feature_names(model):
    if hasattr(model, 'named_steps'):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, 'feature_names_in_'):
            return final_step.feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    return breast_cancer_dataset.feature_names

feature_names = get_feature_names(model)

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f'feature_{i}']) for i in range(len(feature_names))]
    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)[0]
    proba = model.predict_proba(features_array)[0][prediction] * 100 if hasattr(model, 'predict_proba') else None
    result = "Benign (Jinak)" if prediction == 1 else "Malignant (Ganas)"

    return render_template('result.html', result=result, proba=proba)

if __name__ == "__main__":
    app.run(debug=True)
