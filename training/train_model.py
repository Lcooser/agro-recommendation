import pandas as pd
import joblib
from xgboost import XGBClassifier
from feature_engineering import load_dataset

df = load_dataset()

X = df.drop("comprou", axis=1)
y = df["comprou"]


model = XGBClassifier()

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Modelo treinado")
