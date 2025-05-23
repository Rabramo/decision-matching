import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Diretórios
DADOS_BRUTOS = os.path.join('dados', 'brutos')
DADOS_PROCESSADOS = os.path.join('dados', 'processados')
os.makedirs(DADOS_PROCESSADOS, exist_ok=True)

# Funções de ingestão (reaproveita lógica de src/ingestao.py)
def carregar_vagas(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    registros = []
    for vid, vaga in dados.items():
        registro = {'id_vaga': vid}
        if isinstance(vaga, dict):
            for chave, val in vaga.items():
                registro[chave] = val
        registros.append(registro)
    return pd.DataFrame(registros)


def carregar_candidatos(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    registros = []
    for codigo, info in dados.items():
        registro = {'codigo': codigo}
        registro.update(info or {})
        registros.append(registro)
    return pd.DataFrame(registros)

# Função de limpeza de texto
def limpar_texto(texto: str) -> str:
    t = str(texto).lower()
    t = re.sub(r"http\S+|www\.[^\s]+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 1. Carrega e limpa DataFrames
vagas_df = carregar_vagas(os.path.join(DADOS_BRUTOS, 'vagas.json'))
candidatos_df = carregar_candidatos(os.path.join(DADOS_BRUTOS, 'candidatos.json'))

# Detecta coluna de texto em vagas
if 'titulo' in vagas_df.columns:
    campo_vaga = 'titulo'
else:
    cols_v = vagas_df.select_dtypes(include='object').columns.tolist()
    campo_vaga = next((c for c in cols_v if c != 'id_vaga'), None)
    if not campo_vaga:
        raise ValueError(f"Nenhuma coluna de texto encontrada em vagas. Colunas: {vagas_df.columns.tolist()}")
print(f"Usando '{campo_vaga}' como campo de texto para vagas")

# Detecta coluna de texto em candidatos
if 'cv_pt' in candidatos_df.columns:
    campo_cv = 'cv_pt'
else:
    cols_c = candidatos_df.select_dtypes(include='object').columns.tolist()
    campo_cv = next((c for c in cols_c if c != 'codigo'), None)
    if not campo_cv:
        raise ValueError(f"Nenhuma coluna de texto encontrada em candidatos. Colunas: {candidatos_df.columns.tolist()}")
print(f"Usando '{campo_cv}' como campo de texto para candidatos")

# Adiciona colunas limpas
vagas_df['titulo_limpo'] = vagas_df[campo_vaga].fillna('').apply(limpar_texto)
candidatos_df['cv_limpo'] = candidatos_df[campo_cv].fillna('').apply(limpar_texto)

# 2. Salva DataFrames limpos em Parquet
vagas_path = os.path.join(DADOS_PROCESSADOS, 'vagas_cleaned.parquet')
vagas_df.to_parquet(vagas_path, index=False)
print(f"Vagas com texto limpo salvas em: {vagas_path}")
candidatos_path = os.path.join(DADOS_PROCESSADOS, 'candidatos_cleaned.parquet')
candidatos_df.to_parquet(candidatos_path, index=False)
print(f"Candidatos com texto limpo salvos em: {candidatos_path}")

# 3. Vetorização TF-IDF e salvamento de artefatos
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
mat_v = vectorizer.fit_transform(vagas_df['titulo_limpo'])
mat_c = vectorizer.transform(candidatos_df['cv_limpo'])

vec_path = os.path.join(DADOS_PROCESSADOS, 'tfidf_vectorizer.joblib')
joblib.dump(vectorizer, vec_path)
print(f"Vectorizer TF-IDF salvo em: {vec_path}")

v_npz = os.path.join(DADOS_PROCESSADOS, 'tfidf_vagas.npz')
c_npz = os.path.join(DADOS_PROCESSADOS, 'tfidf_candidatos.npz')
sparse.save_npz(v_npz, mat_v)
sparse.save_npz(c_npz, mat_c)
print(f"Matriz TF-IDF de vagas salva em: {v_npz}")
print(f"Matriz TF-IDF de candidatos salva em: {c_npz}")
