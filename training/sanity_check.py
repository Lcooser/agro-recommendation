from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.hybrid import hybrid_recommend, predict_probability


def _base_cliente_frame() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv")
    return (
        df.groupby("cliente_id", as_index=False)
        .agg(
            hectares=("hectares", "first"),
            cultura=("cultura", "first"),
            possui_pulverizador=("possui_pulverizador", "max"),
            possui_plantadeira=("possui_plantadeira", "max"),
            possui_colheitadeira=("possui_colheitadeira", "max"),
            possui_trator=("possui_trator", "max"),
            idade_pulverizador=("idade_pulverizador", "max"),
            idade_plantadeira=("idade_plantadeira", "max"),
            idade_colheitadeira=("idade_colheitadeira", "max"),
            regiao=("regiao", "first"),
        )
    )


def _scenario_payloads() -> list[dict]:
    clientes = _base_cliente_frame()

    small = clientes.sort_values("hectares", ascending=True).iloc[0].to_dict()
    large = clientes.sort_values("hectares", ascending=False).iloc[0].to_dict()

    all_machines_df = clientes[
        (clientes["possui_pulverizador"] == 1)
        & (clientes["possui_plantadeira"] == 1)
        & (clientes["possui_colheitadeira"] == 1)
    ]
    all_machines = (
        all_machines_df.sort_values("hectares", ascending=False).iloc[0].to_dict()
        if not all_machines_df.empty
        else large
    )

    return [
        {"name": "cliente_grande_real", "payload": large},
        {"name": "cliente_pequeno_real", "payload": small},
        {"name": "cliente_todas_maquinas_real", "payload": all_machines},
    ]


def run():
    for item in _scenario_payloads():
        name = item["name"]
        payload = item["payload"]
        cid = int(payload["cliente_id"])
        print(f"\n--- {name} (cliente_id={cid}) ---")
        ranking = hybrid_recommend(payload, cliente_id=cid, top_n=3)
        for rank, rec in enumerate(ranking, start=1):
            print(
                f"{rank}. {rec['produto']} | final={rec['score_final']:.3f} | "
                f"ml={rec['ml_score']:.3f} sim={rec['similarity_score']:.3f} "
                f"collab={rec['collaborative_score']:.3f}"
            )

        payload_prob = dict(payload)
        payload_prob["produto"] = "plantadeira"
        print(f"prob_plantadeira={predict_probability(payload_prob):.3f}")


if __name__ == "__main__":
    run()
