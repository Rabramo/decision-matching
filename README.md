Acesse o MVP de [Decision Matching](https://decision-matching-sy3nsdifq8pgthy7x3muiy.streamlit.app/)

Acesse o [video de digulgação](https://youtu.be/I3d99DVugEg)

[Decision Matching](https://decision-matching-sy3nsdifq8pgthy7x3muiy.streamlit.app/)

Este MVP (Minimum Viable Product) foi desenvolvido durante Datathon promovido pela FIAP/Alura.
É o trabalho final para o curso de pós-graduação lato sensu, Data Analytics, Pós Tech - 6DTAT.

Objetivo do produto proposto

O produto  visa facilitar o processo de recrutamento, automatizando o matching entre vagas disponíveis e currículos de candidatos. Utilizando técnicas de processamento de texto (TF-IDF) e similaridade de cosseno, o sistema gera um ranking dos perfis mais adequados para cada vaga.

Estrutura do Repositório

decision-matching/
├── dados/
│   ├── brutos/           # JSONs originais (vagas.json, candidatos.json)
│   └── processados/      # Artefatos pré-computados (JSON, Parquet, NPZ, Joblib)
├── scripts/
│   └── precompute.py     # Gera arquivos processados e artefatos de TF-IDF
├── src/
│   ├── ingestao.py       # Carregamento e merge de dados
│   ├── matching.py       # Pipeline de pré-processamento e ranking
│   └── artifatos.py      # Funções de leitura de artefatos pré-computados
├── .gitignore            # Regras de versionamento
├── requirements.txt      # Dependências do projeto
├── README.md             # Documentação geral
└──  streamlit_app.py     # Front-end interativo em Streamlit

Instalação e Setup

Clonar repositório

git clone git@github.com:Rabramo/decision-matching.git
cd decision-matching

Criar ambiente virtual (recomendado)

python3 -m venv .venv
source .venv/bin/activate (mac)

Instalar dependências

pip install -r requirements.txt

Pré-computação de Artefatos

Antes de executar o front-end, gere os arquivos processados e artefatos de TF-IDF:

python scripts/precompute.py

Isso criará em dados/processados/:

vagas_cleaned.parquet

candidatos_cleaned.parquet

tfidf_vectorizer.joblib

tfidf_vagas.npz

tfidf_candidatos.npz

Executando o Front-end

Use o Streamlit para iniciar a interface interativa:

No bash, dê o comando streamlit run app/streamlit_app.py

No navegador, selecione uma vaga, ajuste o número de candidatos (Top N) e visualize o ranking gerado.

Fluxo de Implementação

Ingestão de Dados: src/ingestao.py lê JSONs e transforma em DataFrames.

Pré-processamento: limpeza de texto (limpar_texto) em src/matching.py.

Vetorização: TF-IDF sobre descrições de vaga e currículos.

Matching: similaridade de cosseno para ranqueamento.

Front-end: Streamlit carrega artefatos, exibe vaga e resultados.

Contribuindo

Faça um fork do projeto.

Crie uma branch para sua feature: git checkout -b feature-nome

Faça commits das suas alterações: git commit -m "Adiciona feature X"

Envie para a branch remota: git push origin feature-nome

Abra um Pull Request.

