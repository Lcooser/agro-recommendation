from pathlib import Path

import pandas as pd


def load_clients() -> pd.DataFrame:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    df = pd.read_csv(dataset_path)
    for col in [
        "idade_pulverizador",
        "idade_plantadeira",
        "idade_colheitadeira",
        "visitas_ultimos_6_meses",
        "num_oportunidades",
        "valor_oportunidades",
        "maquinas_por_hectare",
        "crescimento_area",
    ]:
        if col not in df.columns:
            df[col] = 0

    profiles = (
        df.groupby("cliente_id", as_index=False)
        .agg(
            hectares=("hectares", "first"),
            regiao=("regiao", "first"),
            cultura=("cultura", "first"),
            possui_pulverizador=("possui_pulverizador", "max"),
            possui_plantadeira=("possui_plantadeira", "max"),
            possui_colheitadeira=("possui_colheitadeira", "max"),
            possui_trator=("possui_trator", "max"),
            idade_pulverizador=("idade_pulverizador", "max"),
            idade_plantadeira=("idade_plantadeira", "max"),
            idade_colheitadeira=("idade_colheitadeira", "max"),
            visitas_ultimos_6_meses=("visitas_ultimos_6_meses", "max"),
            num_oportunidades=("num_oportunidades", "max"),
            valor_oportunidades=("valor_oportunidades", "max"),
            maquinas_por_hectare=("maquinas_por_hectare", "max"),
            crescimento_area=("crescimento_area", "max"),
        )
    )

    return pd.get_dummies(profiles, columns=["regiao", "cultura"])
