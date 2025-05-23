import os
import pandas as pd
import joblib
import scipy.sparse as sparse

# Pasta onde os artefatos processados estão armazenados
dados_processados = os.path.join('dados', 'processados')

# Caminhos para cada artefato
CAMINHO_VAGAS_PARQUET = os.path.join(dados_processados, 'vagas_cleaned.parquet')
CAMINHO_CANDIDATOS_PARQUET = os.path.join(dados_processados, 'candidatos_cleaned.parquet')
CAMINHO_VETORIZADOR = os.path.join(dados_processados, 'tfidf_vectorizer.joblib')
CAMINHO_TFIDF_VAGAS = os.path.join(dados_processados, 'tfidf_vagas.npz')
CAMINHO_TFIDF_CANDIDATOS = os.path.join(dados_processados, 'tfidf_candidatos.npz')


def carregar_vagas_parquet() -> pd.DataFrame:
    """
    Carrega DataFrame de vagas limpas em Parquet.
    """
    return pd.read_parquet(CAMINHO_VAGAS_PARQUET)


def carregar_candidatos_parquet() -> pd.DataFrame:
    """
    Carrega DataFrame de candidatos limpos em Parquet.
    """
    return pd.read_parquet(CAMINHO_CANDIDATOS_PARQUET)


def carregar_vetorizador() -> joblib:
    """
    Carrega o objeto TfidfVectorizer serializado.
    """
    return joblib.load(CAMINHO_VETORIZADOR)


def carregar_matrizes_tfidf():
    """
    Carrega matrizes TF-IDF (vagas e candidatos) em formato esparso.

    Retorna:
        mat_vagas: scipy.sparse.csr_matrix
        mat_candidatos: scipy.sparse.csr_matrix
    """
    mat_vagas = sparse.load_npz(CAMINHO_TFIDF_VAGAS)
    mat_candidatos = sparse.load_npz(CAMINHO_TFIDF_CANDIDATOS)
    return mat_vagas, mat_candidatos
