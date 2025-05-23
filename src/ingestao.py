#%%
import json
import pandas as pd

def carregar_vagas(caminho: str) -> pd.DataFrame:

    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    registros = []
    for vid, vaga in dados.items():
        # Inicia o registro com o ID da vaga
        registro = {'id_vaga': vid}
        # Se o objeto for um dict, “achata” todas as suas chaves
        if isinstance(vaga, dict):
            for chave, valor in vaga.items():
                registro[chave] = valor
        registros.append(registro)

    return pd.DataFrame(registros)


def carregar_candidatos(caminho: str) -> pd.DataFrame:
  
    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    registros = []
    for codigo, info in dados.items():
        registro = {'codigo': codigo}
        registro.update(info or {})
        registros.append(registro)

    return pd.DataFrame(registros)


def merge_vagas_candidatos(
    df_vagas: pd.DataFrame,
    df_candidatos: pd.DataFrame
) -> pd.DataFrame:

    return df_vagas.merge(df_candidatos, how='cross')


#%%   SOMENTE TESTES
# Execução de teste e persistência em Parquet
def main():
    # Carrega ambos DataFrames
    vagas = carregar_vagas('dados/brutos/vagas.json')
    candidatos = carregar_candidatos('dados/brutos/candidatos.json')

    print(f"Vagas carregadas: {vagas.shape}")
    print(f"Candidatos carregados: {candidatos.shape}")

    # Merge cartesiano e salvamento
    df_merged = merge_vagas_candidatos(vagas, candidatos)
    print(f"Merged shape: {df_merged.shape}")

    # Garante existência de pasta processada
    os.makedirs('dados/processados', exist_ok=True)
    caminho_parquet = 'dados/processados/df_merged.parquet'
    df_merged.to_parquet(caminho_parquet, index=False)
    print(f"DataFrame mesclado salvo em: {caminho_parquet}")


if __name__ == '__main__':
    main()


# %%
