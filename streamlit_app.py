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
    carregar_candidatos,
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
    vagas = carregar_vagas()
    candidatos = carregar_candidatos()
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
pagina = st.sidebar.radio("", ["Matching", "Sobre o score", "Video"])

if pagina == "Matching":
    # Título
    st.markdown("<h1 class='stTitle'>Dashboard de Matching (MVP)</h1>", unsafe_allow_html=True)

    # Seletor de vaga
    def format_vaga(idx):
        v = vagas.iloc[idx]
        codigo = v.get('id_vaga', vagas.index[idx])
        titulo = v.get('titulo_vaga', '')
        return f"{codigo} – {titulo}"

    vaga_idx = st.selectbox(
        "Selecione a vaga:",
        options=vagas.index,
        format_func=format_vaga

    )
    # Top N candidatos
    top_n = st.slider("Top N candidatos", 1, 20, 5)

   
    # Cálculo de similaridade sob demanda
    vetor_vaga = mat_v[vaga_idx]
    scores = cosine_similarity(vetor_vaga, mat_c)[0]
    idxs_top = np.argsort(scores)[::-1][:top_n]
    df_top = candidatos.iloc[idxs_top].copy()
    df_top['score'] = scores[idxs_top]
    df_top['codigo'] = candidatos['infos_basicas.codigo_profissional']
    df_top['nome'] = candidatos['infos_basicas.nome']


    # Exibe resultados
    st.markdown(f"<h2 class='stSubheader'>Top {top_n} Candidatos</h2>", unsafe_allow_html=True)
    df_exibir = df_top[['codigo', 'nome', 'score']].reset_index(drop=True)
    st.dataframe(df_exibir, use_container_width=True)

    # Rodapé
    st.markdown("<div class='stMarkdown'>Powered by Decision Matching</div>", unsafe_allow_html=True)

elif pagina == "Sobre o score":
    st.markdown("<h1 class='stTitle'>Como Funciona a Pontuação</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stMarkdown'>
- O **score** traduz o grau de compatibilidade entre o perfil da vaga e o currículo do candidato, capturando semelhanças semânticas de forma quantitativa.<br>
- Cada documento é convertido em um vetor ponderado com **TF-IDF**, amplificando termos-chave que diferenciam perfis e exigências.<br>
- Usamos **similaridade de cosseno** para comparar o vetor da vaga selecionada com o de cada candidato, inferindo proximidade e relevância.<br>
- O score varia de **0** (sem termos relevantes em comum) a **1** (textos quase idênticos), permitindo um ranking claro e confiável.<br>
- Candidatos são ordenados do maior para o menor score, destacando imediatamente os melhores potenciais “matches”.<br>
- Este produto é um **MVP** robusto, já inscrito em um concurso de soluções inovadoras de recrutamento, oferecendo rapidez, escalabilidade e precisão num só clique.<br>
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
    st.video("https://youtu.be/I3d99DVugEg")





# %%
