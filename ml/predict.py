import pandas as pd
import joblib

# Load trained model
model = joblib.load("ml/model.pkl")

# Input values
rainfall = float(input("Enter rainfall: "))
soil_moisture = float(input("Enter soil moisture: "))
slope = float(input("Enter slope: "))
elevation = float(input("Enter elevation: "))
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))
previous_landslide = int(input("Previous landslide? (0 = No, 1 = Yes): "))

# Create input data
data = pd.DataFrame([[
    rainfall,
    soil_moisture,
    slope,
    elevation,
    temperature,
    humidity,
    previous_landslide
]], columns=[
    "rainfall",
    "soil_moisture",
    "slope",
    "elevation",
    "temperature",
    "humidity",
    "previous_landslide"
])

# Prediction
prediction = model.predict(data)[0]

# Probability
probability = model.predict_proba(data)[0][1]

# Risk score
risk_score = round(probability * 100, 2)

# Risk level
if risk_score < 25:
    risk_level = "LOW"
elif risk_score < 50:
    risk_level = "MEDIUM"
elif risk_score < 75:
    risk_level = "HIGH"
else:
    risk_level = "CRITICAL"

print("\n--- Landslide Risk Prediction ---")
print("Prediction:", prediction)
print("Risk Score:", risk_score, "/ 100")
print("Risk Level:", risk_level)