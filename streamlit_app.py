
import os
import sys
import streamlit as st
st.set_page_config(page_title="Decision Matching", layout="wide")
from pathlib import Path
raiz = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(raiz))
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.artefatos import (
    carregar_vagas,
    carregar_candidatos_parquet,
    carregar_matrizes_tfidf,
    carregar_vetorizador
)


# === CSS customizado ===
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    color: #333333;
}
.stTitle { font-size: 18px !important; font-weight: bold; color: #3366CC; }
.stSubheader { font-size: 14px !important; font-weight: bold; color: #3366CC; }
.stMarkdown { font-size: 14px; line-height: 1.6; }
.stButton>button { background-color: #FF7F0E !important; color: white !important; font-size: 16px !important; font-weight: bold !important; border-radius: 8px !important; }
.main .block-container { max-width: 1400px; }
</style>
""", unsafe_allow_html=True)

# === Carregamento de dados e artefatos ===
@st.cache_data
def carregar_dados():
    vagas = carregar_vagas_parquet()
    candidatos = carregar_candidatos_parquet()
    return vagas, candidatos

@st.cache_resource
def carregar_artefatos():
    vetorizer = carregar_vetorizador()
    mat_v, mat_c = carregar_matrizes_tfidf()
    return vetorizer, mat_v, mat_c

vagas, candidatos = carregar_dados()
vetorizador, mat_v, mat_c = carregar_artefatos()

# === Sidebar de navegação ===
st.sidebar.title("Menu")
pagina = st.sidebar.radio("", ["Matching", "Sobre a Pontuação", "Video"])

if pagina == "Matching":
    # Título
    st.markdown("<h1 class='stTitle'>Dashboard de Matching</h1>", unsafe_allow_html=True)

    # Seletor de vaga
    def format_vaga(idx):
        v = vagas.iloc[idx]
        id_vaga = v.get('id_vaga', '')
        titulo_completo = v.get('titulo', '') or v.get('perfil_vaga', {}).get('nome', '')
        return f"{id_vaga} – {titulo_completo}"

    # Dropdown para seleção de vaga
    vaga_idx = st.selectbox(
        "Selecione a vaga:",
        options=vagas.index,
        format_func=format_vaga
    )
    # Top N candidatos
    top_n = st.slider("Top N candidatos", 1, 20, 5)

    # Detalhes da vaga
    vaga = vagas.iloc[vaga_idx]
    st.markdown("<h2 class='stSubheader'>Detalhes da Vaga</h2>", unsafe_allow_html=True)
    cliente      = vaga.get('informacoes_basicas', {}).get('cliente', '')
    titulo       = vaga.get('informacoes_basicas', {}).get('titulo_vaga', vaga.get('titulo', ''))
    tipo         = vaga.get('informacoes_basicas', {}).get('tipo_contratacao', vaga.get('modalidade', ''))
    cidade       = vaga.get('perfil_vaga', {}).get('cidade', vaga.get('cidade', ''))
    competencias = vaga.get('perfil_vaga', {}).get('competencia_tecnicas_e_comportamentais', '')
    observacoes  = vaga.get('perfil_vaga', {}).get('demais_observacoes', '')

    st.markdown(f"""
    <div class='stMarkdown'>
        <p><strong>Cliente:</strong> {cliente}</p>
        <p><strong>Título:</strong> {titulo}</p>
        <p><strong>Tipo de Contratação:</strong> {tipo}</p>
        <p><strong>Cidade:</strong> {cidade}</p>
        <p><strong>Competências:</strong> {competencias}</p>
        <p><strong>Observações:</strong> {observacoes}</p>
    </div>
    """, unsafe_allow_html=True)

    # Cálculo de similaridade sob demanda
    vetor_vaga = mat_v[vaga_idx]
    scores = cosine_similarity(vetor_vaga, mat_c)[0]
    idxs_top = np.argsort(scores)[::-1][:top_n]
    df_top = candidatos.iloc[idxs_top].copy()
    df_top['pontuacao'] = scores[idxs_top]

    # Extrai nome do candidato (achatar json)
    df_top['nome'] = df_top['infos_basicas'].apply(
        lambda d: d.get('nome', '') if isinstance(d, dict) else ''
    )

    # Exibe resultados
    st.markdown(f"<h2 class='stSubheader'>Top {top_n} Candidatos</h2>", unsafe_allow_html=True)
    df_exibir = df_top[['codigo', 'nome', 'pontuacao']].reset_index(drop=True)
    st.dataframe(df_exibir, use_container_width=True)

    # Rodapé
    st.markdown("<div class='stMarkdown'>Powered by Decision Matching</div>", unsafe_allow_html=True)

elif pagina == "Sobre a Pontuação":
    st.markdown("<h1 class='stTitle'>Como Funciona a Pontuação</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stMarkdown'>
    - A **pontuação** mede a similaridade entre o texto da vaga e o texto do currículo.<br>
    - Usamos **TF-IDF** para transformar cada documento em um vetor ponderado.<br>
    -  Calculamos a **similaridade de cosseno** entre o vetor da vaga selecionada e cada vetor de candidato.<br>
    - O valor varia de 0 (sem termos em comum) a 1 (textos idênticos).<br>
    - Os candidatos são ordenados pela pontuação do mais alto para o mais baixo.<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='stMarkdown'>Desenvolvido por Rogério Abramo Alves Pretti</div>", unsafe_allow_html=True)

# === Sobre ===
else:
    st.markdown("<h1 class='stTitle'>Bem-vindo ao Decision Matching</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Sobre o App:**
    Este sistema faz o *matching* entre vagas e currículos usando TF-IDF + similaridade de cosseno.

**Criador:**  
Rogério Abramo Alves Pretti  
""", unsafe_allow_html=True)
    # Exemplo de como incorporar um vídeo (substitua o URL abaixo pelo real)
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
