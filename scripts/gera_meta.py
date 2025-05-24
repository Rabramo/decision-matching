#%%
import json
from pathlib import Path

# Diretórios base do projeto
RAIZ_PROJETO    = Path(__file__).parent.parent.resolve()
BRUTOS_DIR      = RAIZ_PROJETO / 'dados' / 'brutos'
PROCESSADOS_DIR = RAIZ_PROJETO / 'dados' / 'processados'


def generate_meta(
    caminho_json: Path,
    colunas_meta: list[str],
    id_field: str,
    saida_meta: Path
) -> None:

    # Carrega dados brutos
    with open(caminho_json, encoding='utf-8') as f:
        raw = json.load(f)

    meta: list[dict] = []
    for _id, registro in raw.items():
        item: dict = {id_field: _id}
        for key in colunas_meta:
            # busca no nível principal
            val = registro.get(key)
            # se não achar, tenta subníveis comuns
            if val is None and isinstance(registro, dict):
                for sub in ('informacoes_basicas', 'perfil_vaga'):
                    val = registro.get(sub, {}).get(key)
                    if val is not None:
                        break
            item[key] = val or ""
        meta.append(item)

    # Garante existência da pasta de saída
    PROCESSADOS_DIR.mkdir(parents=True, exist_ok=True)
    # Grava JSON formatado
    with open(saida_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Gerado com sucesso: {saida_meta}")


if __name__ == '__main__':
    # === VAGAS ===
    caminho_vagas = BRUTOS_DIR / 'vagas.json'
    saida_vagas   = PROCESSADOS_DIR / 'vagas_meta.json'
    colunas_vagas = [
        'cliente', 'titulo_vaga', 'tipo_contratacao', 'cidade',
        'competencias', 'demais_observacoes'
    ]
    generate_meta(caminho_vagas, colunas_vagas, 'id_vaga', saida_vagas)

    # === CANDIDATOS ===
    caminho_candidatos = BRUTOS_DIR / 'candidatos.json'
    saida_candidatos   = PROCESSADOS_DIR / 'candidatos_meta.json'
    colunas_candidatos = [
        'cv_pt', 'infos_basicas.objetivo_profissional', 'infos_basicas.codigo_profissional',
        'infos_basicas.local', 'infos_basicas.nome', 'informacoes_pessoais.nome',
        'informacoes_pessoais.cpf', 'informacoes_pessoais.data_nascimento',
        'informacoes_pessoais.sexo', 'informacoes_pessoais.estado_civil',
        'informacoes_pessoais.pcd', 'informacoes_profissionais.titulo_profissional',
        'informacoes_profissionais.area_atuacao',
        'informacoes_profissionais.conhecimentos_tecnicos', 'informacoes_profissionais.certificacoes',
        'informacoes_profissionais.outras_certificacoes', 'informacoes_profissionais.remuneracao',
        'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
        'formacao_e_idiomas.nivel_ingles', 'formacao_e_idiomas.nivel_espanhol',
        'formacao_e_idiomas.outro_idioma', 'formacao_e_idiomas.instituicao_ensino_superior',
        'formacao_e_idiomas.cursos', 'formacao_e_idiomas.ano_conclusao',
        'informacoes_profissionais.qualificacoes', 'informacoes_profissionais.experiencias',
        'formacao_e_idiomas.outro_curso', 'cargo_atual.id_ibrati'
    ]
    generate_meta(caminho_candidatos, colunas_candidatos, 'id_candidato', saida_candidatos)
