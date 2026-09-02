from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("ml/model.pkl")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([[
        data["rainfall"],
        data["soil_moisture"],
        data["slope"],
        data["elevation"],
        data["temperature"],
        data["humidity"],
        data["previous_landslide"]
    ]], columns=[
        "rainfall",
        "soil_moisture",
        "slope",
        "elevation",
        "temperature",
        "humidity",
        "previous_landslide"
    ])

    probability = model.predict_proba(input_data)[0][1]

    risk_score = round(probability * 100, 2)

    if risk_score < 25:
        risk_level = "LOW"
    elif risk_score < 50:
        risk_level = "MEDIUM"
    elif risk_score < 75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return jsonify({
        "risk_score": risk_score,
        "risk_level": risk_level
    })


if __name__ == "__main__":
    app.run(debug=True)