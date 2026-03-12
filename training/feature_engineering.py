import pandas as pd
from pathlib import Path

def load_dataset():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    df = pd.read_csv(dataset_path)
    
    df = pd.get_dummies(df, columns=["cultura", "regiao", "produto"])
    
    return df
