import json
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import joblib
import scipy.sparse as sparse
import streamlit as st

# Define a raiz do projeto dinamicamente
BASE_DIR = Path(__file__).resolve().parent.parent
BRUTOS_DIR = BASE_DIR / "dados" / "brutos"
PROCESSADOS_DIR = BASE_DIR / "dados" / "processados"
# Garante existência da pasta de processados
PROCESSADOS_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos para cada artefato
CAMINHO_VAGAS_META        = PROCESSADOS_DIR / "vagas_meta.json"
CAMINHO_CANDIDATOS_META   = PROCESSADOS_DIR / "candidatos_meta.json"
CAMINHO_CANDIDATOS_PARQ   = PROCESSADOS_DIR / "candidatos_cleaned.parquet"
CAMINHO_VETORIZADOR       = PROCESSADOS_DIR / "tfidf_vectorizer.joblib"
CAMINHO_TFIDF_VAGAS       = PROCESSADOS_DIR / "tfidf_vagas.npz"
CAMINHO_TFIDF_CANDIDATOS  = PROCESSADOS_DIR / "tfidf_candidatos.npz"

@st.cache_data
def carregar_vagas() -> pd.DataFrame:

    # Gera JSON a partir do bruto, se não existir
    if not CAMINHO_VAGAS_META.exists():
        raw_path = BRUTOS_DIR / "vagas.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw.values())
        df.to_json(CAMINHO_VAGAS_META, orient="records", force_ascii=False, indent=2)
        return df

    # Lê JSON pré-gerado
    with open(CAMINHO_VAGAS_META, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def carregar_candidatos() -> pd.DataFrame:

    # Se já existe Parquet limpo, lê direto
    if CAMINHO_CANDIDATOS_PARQ.exists():
        return pd.read_parquet(CAMINHO_CANDIDATOS_PARQ)

    # Caso contrário, gera JSON meta se não existir
    if not CAMINHO_CANDIDATOS_META.exists():
        raw_path = BRUTOS_DIR / "candidatos.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # raw é dict id->dado; normaliza valores em DataFrame
        df = pd.json_normalize(list(raw.values()))
        df.to_json(CAMINHO_CANDIDATOS_META, orient="records", force_ascii=False, indent=2)
    else:
        with open(CAMINHO_CANDIDATOS_META, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    # Grava Parquet para próximas execuções
    df.to_parquet(CAMINHO_CANDIDATOS_PARQ)
    return df

@st.cache_data
def carregar_vetorizador() -> Any:

    return joblib.load(CAMINHO_VETORIZADOR)

@st.cache_data
def carregar_matrizes_tfidf() -> Tuple[Any, Any]:

    mat_vagas = sparse.load_npz(CAMINHO_TFIDF_VAGAS)
    mat_candidatos = sparse.load_npz(CAMINHO_TFIDF_CANDIDATOS)
    return mat_vagas, mat_candidatos