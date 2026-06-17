import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("hyundai_regression_dataset.csv")

X = data[["year", "engine_size", "fuel_efficiency", "horsepower"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=123
    ))
])

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

joblib.dump(modelo, "modelo_hyundai.joblib")

print("Modelo guardado correctamente.")