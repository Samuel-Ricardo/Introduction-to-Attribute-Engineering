# Introduction to Attribute Engineering

Projeto prático de **Engenharia de Atributos** para a disciplina **ES510 - Introdução à IA (UFPE)**.

Este repositório implementa, em Python, um fluxo completo de preparação e seleção de atributos no dataset de crédito, incluindo:

- análise de valores ausentes;
- imputação por moda;
- seleção de atributos por filtros estatísticos;
- análise de correlação;
- métodos Wrapper e Embedded;
- remoção de variáveis;
- criação de atributos polinomiais.

## 1. Contexto

O projeto segue o laboratório de engenharia de atributos e tem como objetivo transformar o conjunto de dados bruto em uma base mais informativa para modelagem, reduzindo ruído e redundância.

Arquivos-base da atividade:

- enunciado: `[ES510]_Lab1.pdf`
- dataset: `[ES_510]_Lab01/credit.csv`
- notebook original: `[ES_510]_Lab01/lab1.ipynb`

Implementação principal no projeto:

- script de execução por células: `src/model.py`

## 2. Estrutura do Repositório

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── [ES510]_Lab1.pdf
├── [ES_510]_Lab01/
│   ├── credit.csv
│   └── lab1.ipynb
├── src/
│   └── model.py
└── ai_hello/               # ambiente virtual local (ignorado no git)
```

## 3. Tecnologias e Dependências

Dependências declaradas em `requirements.txt`:

- `pandas`
- `missingno`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## 4. Pré-requisitos

- Python 3.10+ (recomendado 3.11)
- VS Code com extensão Python (e opcionalmente Jupyter para execução por células)
- Git (opcional, para versionamento)

## 5. Setup do Ambiente

### 5.1 Criar e ativar ambiente virtual

No PowerShell, na raiz do projeto:

```powershell
python -m venv ai_hello
.\ai_hello\Scripts\Activate.ps1
```

### 5.2 Instalar dependências

```powershell
pip install -r requirements.txt
```

## 6. Como Executar

### Opção A: VS Code (recomendado)

O arquivo `src/model.py` foi organizado com `#%%`, então cada bloco funciona como uma célula.

Passos:

1. Abra `src/model.py` no VS Code.
2. Ative o ambiente virtual `ai_hello`.
3. Execute célula por célula com **Run Cell**.
4. Siga a ordem do arquivo para preservar variáveis (`df`, `X`, `corr_matrix`, etc.).

### Opção B: Terminal (script inteiro)

Na raiz do projeto:

```powershell
cd src
python model.py
```

Observação: ao executar como script tradicional, os blocos `#%%` são apenas comentários e não interrompem a execução.

## 7. Pipeline Implementado

Abaixo está o fluxo implementado em `src/model.py`.

### 7.1 Carregamento dos dados

- Lê `../[ES_510]_Lab01/credit.csv` com pandas.
- Inspeciona as primeiras linhas (`df.head()`).

### 7.2 Diagnóstico de valores ausentes

- Visualização com `missingno.matrix(df)`.
- Cálculo da taxa de missing por coluna (`df_missing`).

### 7.3 Imputação por moda

- Lista de colunas com missing:
  - `Title`, `Industry`, `House_State`, `Nation`, `Marriage_State`, `Highest Education`, `Duty`.
- Preenchimento com `mode()[0]`.

### 7.4 Crosstab (House_State x Target)

- Cria tabela de contingência (`pd.crosstab`).
- Calcula proporção por linha (`cross_table_rowpct`).

### 7.5 Teste Qui-Quadrado (Filter)

- Define:
  - `X = df.drop('Target', axis=1)`
  - `y = df['Target']`
- Separa variáveis categóricas em `X_category`.
- Aplica `chi2_test(X_category, y)` (alias de `sklearn.feature_selection.chi2`).
- Ordena as features por score em `ls`.

### 7.6 Correlação entre variáveis contínuas

- Define variáveis nominais em `nominal_features`.
- Gera lista de variáveis numéricas por exclusão (`numerical_features`).
- Cria matriz de correlação Spearman (`corr_matrix`).
- Plota heatmap com `seaborn` (`cmap='magma'`).

### 7.7 Pares altamente correlacionados

- Varre `corr_matrix` e coleta pares com correlação `>= 0.8` em `cols_pair`.
- Evita pares duplicados invertidos (A,B) e (B,A).

### 7.8 Método Wrapper (RFE)

- Modelo base: `LogisticRegression(max_iter=1000, solver='liblinear')`.
- `RFE` com `n_features_to_select=10`.
- Exibe:
  - número de atributos selecionados;
  - máscara de suporte;
  - ranking;
  - modelo final;
  - nomes das features selecionadas.

### 7.9 Método Embedded (Random Forest)

- Treina `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Calcula `feature_importances_`.
- Ordena em `sorted_feature`.

### 7.10 Remoção de variáveis

- Remove colunas definidas em `del_cols`.
- Gera novo dataframe `df_select`.

### 7.11 Construção de atributos polinomiais

- Usa features base:
  - `Ast_Curr_Bal`, `Age`, `Year_Income`, `Std_Cred_Limit`.
- Aplica `PolynomialFeatures(degree=3)`.
- Cria dataframe expandido `poly_features`.

### 7.12 Correlação com Target nas novas features

- Calcula `poly_features.corr()['Target']`.
- Exibe 5 menores e 5 maiores coeficientes.

## 8. Saídas Esperadas

Durante a execução, você deve obter:

- matriz visual de missing data;
- tabela de contingência e percentuais por linha;
- ranking de variáveis categóricas pelo qui-quadrado;
- heatmap de correlação das variáveis contínuas;
- lista de pares com correlação alta (`cols_pair`);
- seleção de atributos por RFE e importância por Random Forest;
- conjunto polinomial e ranking de correlação com `Target`.

## 9. Problemas Comuns e Soluções

### 9.1 `KeyError` em nomes de colunas

Verifique nomes exatos em `df.columns`.
Exemplos corretos no dataset:

- `Marriage_State`
- `Highest Education`
- `Work_Years`

### 9.2 `NameError: chi2_test is not defined`

Garanta que exista alias antes da chamada:

```python
from sklearn.feature_selection import chi2
chi2_test = chi2
```

### 9.3 `ValueError: Input contains NaN` no qui-quadrado

O `chi2` não aceita NaN. Execute a imputação antes do teste:

```python
for col in missing_col:
    df[col] = df[col].fillna(df[col].mode()[0])
```

### 9.4 Erro de caminho do dataset

Se executar a partir de `src/`, o caminho relativo esperado no script é:

- `../[ES_510]_Lab01/credit.csv`

## 10. Boas Práticas Recomendadas

- Sempre executar as células em ordem.
- Manter random seeds fixos para reprodutibilidade (`random_state=42`).
- Validar missing values após cada transformação.
- Revisar multicolinearidade antes de treinar modelos finais.

## 11. Possíveis Extensões

- adicionar avaliação de modelos (AUC/F1) antes e depois da engenharia de atributos;
- salvar artefatos intermediários (`df_select`, `poly_features`) em `csv/parquet`;
- encapsular o pipeline em funções/classes para reutilização;
- criar testes para validação de schema e qualidade de dados.

## 12. Autor

Projeto acadêmico desenvolvido para prática de engenharia de atributos.

Se quiser, posso complementar este README com:

- badges (Python, versão, status);
- seção de resultados com imagens exportadas;
- instruções de contribuição e convenção de commits.
