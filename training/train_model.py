from pathlib import Path

import joblib
from xgboost import XGBClassifier
from feature_engineering import load_dataset

df = load_dataset()

X = df.drop("comprou", axis=1)
y = df["comprou"]


model = XGBClassifier()

model.fit(X, y)

model_path = Path(__file__).resolve().parent / "model.pkl"
joblib.dump(model, model_path)

print("Modelo treinado")
