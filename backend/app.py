from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)
CORS(app)

MODEL_PATH = "modelo_hyundai.joblib"
DATASET_PATH = "hyundai_regression_dataset.csv"


def entrenar_modelo():
    data = pd.read_csv(DATASET_PATH)

    X = data[["year", "engine_size", "fuel_efficiency", "horsepower"]]
    y = data["price"]

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=123
        ))
    ])

    modelo.fit(X, y)

    joblib.dump(modelo, MODEL_PATH)

    return modelo


if os.path.exists(MODEL_PATH):
    try:
        modelo = joblib.load(MODEL_PATH)
    except Exception:
        modelo = entrenar_modelo()
else:
    modelo = entrenar_modelo()


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
