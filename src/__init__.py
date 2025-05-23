import json
import pandas as pd

def carregar_e_mesclar_dados(
    caminho_vagas: str,
    caminho_prospeccoes: str,
    caminho_candidatos: str
) -> pd.DataFrame:
    """
    Carrega os três JSONs (vagas, prospeccoes, candidatos), normaliza em DataFrames e retorna
    um DataFrame mesclado com informações de vaga e dados completos do candidato.
    Colunas principais:
      - id_vaga, titulo, modalidade,
      - codigo, campos do candidato,
      - situacao_candidato (do prospect)
    """
    # Carrega JSONs
    with open(caminho_vagas, 'r', encoding='utf-8') as f:
        dados_vagas = json.load(f)
    with open(caminho_prospeccoes, 'r', encoding='utf-8') as f:
        dados_prosp = json.load(f)
    with open(caminho_candidatos, 'r', encoding='utf-8') as f:
        dados_cand = json.load(f)

    # DataFrame de vagas
    df_vagas = pd.DataFrame([
        {'id_vaga': vid, 'titulo': v.get('titulo', ''), 'modalidade': v.get('modalidade', '')}
        for vid, v in dados_vagas.items()
    ])

    # DataFrame de prospeccoes (flat)
    registros = []
    for vid, info in dados_prosp.items():
        for p in info.get('prospects', []):
            registros.append({
                'id_vaga': vid,
                'codigo': p.get('codigo', ''),
                'situacao_candidato': p.get('situacao_candidato', '')
            })
    df_prosp = pd.DataFrame(registros)

    # DataFrame de candidatos
    df_cand = pd.DataFrame([
        {'codigo': codigo, **(cand or {})}
        for codigo, cand in dados_cand.items()
    ])

    # Merge: prospects + vagas
    df_merged = (
        df_prosp
        .merge(df_vagas, on='id_vaga', how='left')
        .merge(df_cand, on='codigo', how='left')
    )
    return df_merged
