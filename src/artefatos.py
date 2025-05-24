import json
from pathlib import Path

import pandas as pd
import joblib
import scipy.sparse as sparse
import streamlit as st

# Define a raiz do projeto dinamicamente
BASE_DIR = Path(__file__).resolve().parent.parent  # assume src/artefatos.py
PROCESSADOS = BASE_DIR / "dados" / "processados"

# Caminhos para cada artefato
CAMINHO_VAGAS_META         = PROCESSADOS / "vagas_meta.json"
CAMINHO_CANDIDATOS_PARQUET = PROCESSADOS / "candidatos_cleaned.parquet"
CAMINHO_VETORIZADOR        = PROCESSADOS / "tfidf_vectorizer.joblib"
CAMINHO_TFIDF_VAGAS        = PROCESSADOS / "tfidf_vagas.npz"
CAMINHO_TFIDF_CANDIDATOS   = PROCESSADOS / "tfidf_candidatos.npz"

@st.cache_data
def carregar_vagas() -> pd.DataFrame:

    with open(CAMINHO_VAGAS_META, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def carregar_candidatos() -> pd.DataFrame:
    """
    Carrega o Parquet de candidatos e retorna um DataFrame.
    """
    return pd.read_parquet(CAMINHO_CANDIDATOS_PARQUET)

@st.cache_data
def carregar_vetorizador() -> joblib:

    return joblib.load(CAMINHO_VETORIZADOR)

@st.cache_data
def carregar_matrizes_tfidf() -> tuple:

    mat_vagas      = sparse.load_npz(CAMINHO_TFIDF_VAGAS)
    mat_candidatos = sparse.load_npz(CAMINHO_TFIDF_CANDIDATOS)
    return mat_vagas, mat_candidatos
