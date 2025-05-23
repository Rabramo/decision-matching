
#%%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.artefatos import (
    carregar_vagas_parquet,
    carregar_candidatos_parquet,
    carregar_matrizes_tfidf
)


def obter_top_candidatos(
    idx_vaga: int,
    top_n: int = 5
) -> pd.DataFrame:
 
    # Carregar dados e matrizes
    vagas = carregar_vagas_parquet()
    candidatos = carregar_candidatos_parquet()
    mat_v, mat_c = carregar_matrizes_tfidf()

    # Vetor da vaga selecionada
    vetor_vaga = mat_v[idx_vaga]

    # Similaridade de cosseno
    scores = cosine_similarity(vetor_vaga, mat_c)[0]

    # Seleciona top_n índices
    idxs_top = scores.argsort()[::-1][:top_n]

    # Pega apenas as linhas desejadas
    resultados = candidatos.iloc[idxs_top].copy()
    resultados['pontuacao'] = scores[idxs_top]
    return resultados

#%%
# Exemplo de execução
def main():
    # Ajuste idx_vaga e top_n conforme necessário
    idx_vaga = 0
    top_n = 5
    top5 = obter_top_candidatos(idx_vaga, top_n)
    print(f"Top {top_n} candidatos para vaga {idx_vaga}:")
    print(top5[['codigo', 'pontuacao']])


if __name__ == '__main__':
    main()
