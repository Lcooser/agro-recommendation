from pathlib import Path

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model_path = Path(__file__).resolve().parents[1] / "training" / "model.pkl"
model = joblib.load(model_path)
expected_features = model.get_booster().feature_names


class ClienteInput(BaseModel):
    cliente_id: int
    produto: str
    hectares: float
    cultura: str
    possui_pulverizador: int
    possui_plantadeira: int
    possui_colheitadeira: int
    possui_trator: int
    regiao: str


@app.post("/recommend")
def recommend(cliente: ClienteInput):
    df = pd.DataFrame([cliente.model_dump()])
    df = pd.get_dummies(df, columns=["cultura", "regiao", "produto"])
    df = df.reindex(columns=expected_features, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    return {"probabilidade_compra": float(prob)}
