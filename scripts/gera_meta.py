#%%
import os, json, pandas as pd
from pathlib import Path


RAIZ_PROJETO = Path(__file__).parent.parent.resolve()


caminho_json = RAIZ_PROJETO / 'dados' / 'brutos' / 'vagas.json'
saida_meta   = RAIZ_PROJETO / 'dados' / 'processados' / 'vagas_meta.json'



with open(caminho_json, encoding='utf-8') as f:
    vagas = json.load(f)


colunas_meta = [
    'cliente',
    'titulo_vaga',
    'tipo_contratacao',
    'cidade',
    'competencias',
    'demais_observacoes'
]

meta = []
for vid, v in vagas.items():
    registro = {'id_vaga': vid}
    for key in colunas_meta:
        # v pode ter a chave no nível superior ou dentro de sub-dicts
        val = v.get(key)
        if val is None:
            # busca em infos_basicas e perfil_vaga
            val = (v.get('informacoes_basicas') or {}).get(key) \
                  or (v.get('perfil_vaga') or {}).get(key) \
                  or ""
        registro[key] = val
    meta.append(registro)

#%%

os.makedirs(os.path.dirname(saida_meta), exist_ok=True)
with open(saida_meta, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Gerado com sucesso:", saida_meta)
# %%
