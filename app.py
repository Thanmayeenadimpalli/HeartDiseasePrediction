from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load("model/heart_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read values from the form
    age = int(request.form["age"])
    sex = request.form["sex"]
    cp = request.form["cp"]
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    thalch = float(request.form["thalch"])
    oldpeak = float(request.form["oldpeak"])
    ca = float(request.form["ca"])
    fbs = request.form["fbs"] == "True"
    restecg = request.form["restecg"]
    exang = request.form["exang"] == "True"
    slope = request.form["slope"]
    thal = request.form["thal"]

    # Create a DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "ca": ca,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "thal": thal
    }])

    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][prediction] * 100

    if prediction == 1:
        result = f"⚠️ High Risk of Heart Disease ({probability:.2f}% confidence)"
    else:
        result = f"✅ Low Risk of Heart Disease ({probability:.2f}% confidence)"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)