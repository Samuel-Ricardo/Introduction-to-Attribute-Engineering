# Projeto 01 - Engenharia de Atributos

Projeto academico da disciplina ES510 (Introducao a IA - UFPE) com foco em engenharia de atributos usando um dataset de credito.

## Objetivo

Implementar um pipeline completo de preparacao e selecao de atributos, passando por:

- analise de valores ausentes;
- imputacao por moda;
- analise de dependencia com qui-quadrado;
- analise de correlacao entre variaveis numericas;
- selecao de atributos por metodos Wrapper e Embedded;
- remocao de colunas e geracao de atributos polinomiais.

## Estrutura

```text
01/
├── README.md
├── requirements.txt
├── [ES_510]_Lab01/
│   ├── credit.csv
│   └── lab1.ipynb
├── src/
│   └── model.py
└── ai_hello/                # venv local
```

## Requisitos

- Python 3.10+ (recomendado 3.11)
- Ambiente virtual ativo
- Dependencias do arquivo requirements.txt

## Setup

No PowerShell, a partir da pasta 01:

```powershell
python -m venv ai_hello
.\ai_hello\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como executar

Opcao 1 (VS Code com celulas):

1. Abra src/model.py.
2. Selecione o interpretador do venv ai_hello.
3. Execute os blocos #%% em ordem.

Opcao 2 (terminal):

```powershell
cd src
python model.py
```

## Saidas esperadas

- visualizacoes de missing data;
- tabela de contingencia e proporcoes;
- ranking de features por qui-quadrado;
- heatmap de correlacao;
- features selecionadas por RFE e RandomForest;
- correlacao das features polinomiais com Target.

## Problemas comuns

- KeyError em colunas: valide nomes com df.columns.
- Erro no caminho do dataset: execute mantendo a estrutura original.
- Erro com NaN no qui-quadrado: garanta imputacao antes do teste.

## Arquivos principais

- src/model.py — pipeline de engenharia de atributos
- src/questoes.py — respostas das 10 questões da seção 4.2.4 do PDF

## Contexto academico

Laboratorio [ES_510]_Lab01 com implementacao em script Python para reproduzir os passos do notebook.
