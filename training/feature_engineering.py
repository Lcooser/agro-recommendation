import pandas as pd
from pathlib import Path


AGE_COLUMNS = ["idade_pulverizador", "idade_plantadeira", "idade_colheitadeira"]
BUSINESS_COLUMNS = [
    "visitas_ultimos_6_meses",
    "num_oportunidades",
    "valor_oportunidades",
    "maquinas_por_hectare",
    "crescimento_area",
]
MACHINE_FLAGS = ["possui_pulverizador", "possui_plantadeira", "possui_colheitadeira", "possui_trator"]


def load_dataset():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    df = pd.read_csv(dataset_path)

    # Backward-compatible defaults when age columns are not present yet.
    for col in AGE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    for col in AGE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in BUSINESS_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "maquinas_por_hectare" in df.columns:
        missing_ratio = (df["maquinas_por_hectare"] <= 0).mean()
        if missing_ratio > 0.5:
            machine_count = sum(df.get(col, 0) for col in MACHINE_FLAGS if col in df.columns)
            hectares_safe = df["hectares"].replace(0, 1)
            df["maquinas_por_hectare"] = machine_count / hectares_safe

    df = pd.get_dummies(df, columns=["cultura", "regiao", "produto"])

    return df
