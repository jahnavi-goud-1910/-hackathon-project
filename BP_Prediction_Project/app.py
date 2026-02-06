from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("hypertension_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Collect form inputs
    gender = int(request.form["gender"])
    age_group = int(request.form["age_group"])
    family = int(request.form["family_history"])
    treatment = int(request.form["treatment"])
    breath = int(request.form["breath"])
    vision = int(request.form["vision"])
    nose = int(request.form["nose"])
    systolic = int(request.form["systolic"])
    diastolic = int(request.form["diastolic"])
    diet = int(request.form["diet"])
    medication = int(request.form["medication"])

    # ML prediction
    features = np.array([[gender, age_group, systolic, diastolic]])
    stage = model.predict(features)[0]
    confidence = round(model.predict_proba(features).max() * 100, 2)

    # Stage mapping
    if stage == 0:
        result = "Normal Blood Pressure"
        advice = "Maintain healthy lifestyle."
        action = "Regular monitoring"
        risk_class = "normal"
    elif stage == 1:
        result = "Stage-1 Hypertension"
        advice = "Lifestyle modification recommended."
        action = "Reduce salt & exercise"
        risk_class = "moderate"
    elif stage == 2:
        result = "Stage-2 Hypertension"
        advice = "Medical consultation required."
        action = "Medication & monitoring"
        risk_class = "high"
    else:
        result = "Hypertensive Crisis"
        advice = "Immediate medical attention needed."
        action = "Emergency care"
        risk_class = "critical"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        advice=advice,
        action=action,
        risk_class=risk_class
    )

if __name__ == "__main__":
    app.run(debug=True)
