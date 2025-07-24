from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model_svm.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    hasil = None
    risiko_ganas = None
    risiko_jinak = None

    if request.method == "POST":
        try:
            features = []
            for i in range(1, 31):
                val = request.form.get(f"f{i}")
                if val.strip() == "":
                    raise ValueError("Semua fitur harus diisi.")
                features.append(float(val))

            arr = np.array([features])
            proba = model.predict_proba(arr)[0]
            pred = model.predict(arr)[0]

            hasil = "Tumor Ganas (Malignant)" if pred == 0 else "Tumor Jinak (Benign)"
            risiko_ganas = f"{proba[0]*100:.2f}%"
            risiko_jinak = f"{proba[1]*100:.2f}%"

        except Exception as e:
            hasil = f"❌ Error: {e}"

    return render_template("index.html", hasil=hasil, risiko_ganas=risiko_ganas, risiko_jinak=risiko_jinak)

if __name__ == "__main__":
    app.run(debug=True)
