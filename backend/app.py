from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

modelo = joblib.load("modelo_hyundai.joblib")

@app.route("/")
def home():
    return jsonify({
        "mensaje": "API de predicción de precio Hyundai funcionando"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    year = int(data["year"])
    engine_size = float(data["engine_size"])
    fuel_efficiency = float(data["fuel_efficiency"])
    horsepower = int(data["horsepower"])

    entrada = pd.DataFrame([{
        "year": year,
        "engine_size": engine_size,
        "fuel_efficiency": fuel_efficiency,
        "horsepower": horsepower
    }])

    prediccion = modelo.predict(entrada)[0]

    return jsonify({
        "precio_predicho": round(float(prediccion), 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)