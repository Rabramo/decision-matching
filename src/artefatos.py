import json
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import joblib
import scipy.sparse as sparse
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
BRUTOS_DIR = BASE_DIR / "dados" / "brutos"
PROCESSADOS_DIR = BASE_DIR / "dados" / "processados"
PROCESSADOS_DIR.mkdir(parents=True, exist_ok=True)

CAMINHO_VAGAS_BRUTOS = BRUTOS_DIR / "vagas.json"
CAMINHO_CANDIDATOS_BRUTOS = BRUTOS_DIR / "candidatos.json"

CAMINHO_VAGAS_PARQ = PROCESSADOS_DIR / "vagas_cleaned.parquet"
CAMINHO_CANDIDATOS_PARQ = PROCESSADOS_DIR / "candidatos_cleaned.parquet"
CAMINHO_VETORIZADOR = PROCESSADOS_DIR / "tfidf_vectorizer.joblib"
CAMINHO_TFIDF_VAGAS = PROCESSADOS_DIR / "tfidf_vagas.npz"
CAMINHO_TFIDF_CANDIDATOS = PROCESSADOS_DIR / "tfidf_candidatos.npz"

@st.cache_data
def carregar_vagas() -> pd.DataFrame:
    with open(CAMINHO_VAGAS_BRUTOS, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for id_vaga, entry in raw.items():
        info = entry.get("informacoes_basicas", {})
        info["id_vaga"] = id_vaga
        records.append(info)
    df = pd.DataFrame(records)
    df.to_parquet(CAMINHO_VAGAS_PARQ, index=False)
    return df

@st.cache_data
def carregar_candidatos() -> pd.DataFrame:
    if CAMINHO_CANDIDATOS_PARQ.exists():
        return pd.read_parquet(CAMINHO_CANDIDATOS_PARQ)
    with open(CAMINHO_CANDIDATOS_BRUTOS, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.json_normalize(list(raw.values()))
    df.to_parquet(CAMINHO_CANDIDATOS_PARQ, index=False)
    return df

@st.cache_data
def carregar_vetorizador() -> Any:
    return joblib.load(CAMINHO_VETORIZADOR)

@st.cache_data
def carregar_matrizes_tfidf() -> Tuple[Any, Any]:
    mat_vagas = sparse.load_npz(CAMINHO_TFIDF_VAGAS)
    mat_candidatos = sparse.load_npz(CAMINHO_TFIDF_CANDIDATOS)
    return mat_vagas, mat_candidatos
