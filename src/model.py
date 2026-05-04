
#%%

import pandas as pd

#LOAD DATASET
df=pd.read_csv("../[ES_510]_Lab01/credit.csv");

df.head()

# %%

import missingno as msno

#VIEW MISSING DATA
msno.matrix(df)

# %%

#CREATE A DATAFRAME WITH THE RATE OF MISSING VALUES PER COLUMN

#df.isnull().sum()   # COUNT NULL VAULES BY COLUMN 
#df.shape[0]         # COUNT ROWS

df_missing = pd.DataFrame(df.isnull().sum()/df.shape[0], columns=['missing_rate']).reset_index()

df_missing.sort_values(by='missing_rate', ascending=False)[:15]


# %%

#FILL WITH MODE (MODA)

missing_col = ['Title', 'Industry', 'House_State', 'Nation', 'Marriage_State', 'Highest Education', 'Duty']

df_missing_2 = pd.DataFrame(df.isnull().sum()/df.shape[0], columns=['missing_rate']).reset_index()
df_missing_2.sort_values(by='missing_rate', ascending=False)[:15]

# %%

# GENERATE CROSS TABLE

cross_table = pd.crosstab(df.House_State, columns = df.Target, margins = True)

cross_table_rowpct = cross_table.div(cross_table['All'], axis = 0)
cross_table_rowpct

# %%

# CHI-SQUARE TEST

for col in missing_col:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop('Target', axis = 1)
y = df['Target']

X_category = df[['Nation', 'Birth_Place', 'Gender', 'Marriage_State', 'Highest Education', 'House_State', 'Work_Years', 'Title', 'Duty', 'Industry']]


# %%

from sklearn.feature_selection import chi2

chi2_test = chi2

#APPLY CHI-SQUARE TEST
# chi2, pval = chi2(X_category, y)

#CREATE DICTIONARY TO STORE THE RESULTS
# dict_feature = {}

#ASSOCIATE EACH ATTRIBUTE TO THE VALUE OF CHI-SQUARE
# for i,j in zip(X_category.columns.values,chi2):
# dict_feature[...] = ...

(chi2_vals, pval) = chi2_test(X_category, y)

# Criar dicionário para armazenar os resultados
dict_feature = {}

# Associar cada atributo ao valor do qui-quadrado
for i, j in zip(X_category.columns.values, chi2_vals):
    dict_feature[i] = j

# Ordenar os atributos pela importância (ordem decrescente)
ls = sorted(dict_feature.items(), key=lambda item: item[1], reverse=True)

# Exibir resultado
ls

# %%

# PASSO 3: ANÁLISE DE CORRELAÇÃO ENTRE VARIÁVEIS CONTÍNUAS

import matplotlib.pyplot as plt
import seaborn as sns

# Definir lista de variáveis categóricas (nominais)
nominal_features = ['Nation', 'Birth_Place', 'Gender', 'Marriage_State',
                    'Highest Education', 'House_State', 'Work_Years',
                    'Unit_Kind', 'Title', 'Occupation', 'Duty', 'Industry']

# Selecionar variáveis numéricas (todas que não estão em nominal_features)
numerical_features = [col for col in X.columns if col not in nominal_features]

# Remover o primeiro elemento da lista (Cust_No ou índice irrelevante)
numerical_features.pop(0)

# Criar subconjunto com variáveis numéricas
X_num = df[numerical_features]

# Calcular matriz de correlação (Spearman)
corr_matrix = X_num.corr(method='spearman')

# Definir tamanho da figura
plt.figure(figsize=(16, 12))

# Plotar mapa de calor
sns.heatmap(corr_matrix, annot=True, fmt='.2g', cmap='magma')
plt.tight_layout()
plt.show()

# %%

# IDENTIFICAR PARES DE VARIÁVEIS COM ALTA CORRELAÇÃO (>= 0.8)

cols_pair = []
for index_ in corr_matrix.index:
    for col_ in corr_matrix.columns:
        if corr_matrix.loc[index_, col_] >= 0.8 and index_ != col_ and (col_, index_) not in cols_pair:
            cols_pair.append((index_, col_))

# Exibir pares de variáveis altamente correlacionadas
cols_pair

# %%

# 3.4 MÉTODO WRAPPER - RECURSIVE FEATURE ELIMINATION (RFE)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Garantir que X esteja livre de NaN para o LogisticRegression
X = df.drop('Target', axis=1).fillna(0)
y = df['Target']

# Definir modelo base
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Aplicar RFE selecionando 10 atributos
x_rfe = RFE(estimator=model, n_features_to_select=10).fit(X, y)

# Exibir número de atributos selecionados
print("Número de atributos selecionados:", x_rfe.n_features_)

# Exibir máscara de seleção (True/False)
print("Suporte (máscara True/False):", x_rfe.support_)

# Exibir ranking dos atributos
print("Ranking dos atributos:", x_rfe.ranking_)

# Exibir modelo final treinado
print("Modelo final:", x_rfe.estimator_)

# Mostrar quais atributos foram selecionados
selected_features = X.columns[x_rfe.support_].tolist()
print("Atributos selecionados pelo RFE:", selected_features)

# %%

# 3.5 MÉTODO EMBEDDED - RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

# Criar modelo Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar modelo
rfc.fit(X, y)

# Obter nomes das colunas
cols = X.columns.tolist()

# Calcular importância dos atributos e ordenar
sorted_feature = sorted(
    zip(map(lambda x: round(x, 4), rfc.feature_importances_), cols),
    reverse=True
)

# Exibir ranking das variáveis
sorted_feature

# %%

# 3.6 REMOÇÃO DE VARIÁVEIS

# Definir lista de variáveis a serem removidas (com base nos resultados anteriores)
del_cols = ['Gender', 'House_State', 'Couple_Year_Income', 'Loan_Curr_Bal',
            'ZX_Max_Credit_Banks', 'ZX_Max_Overdue_Credits',
            'ZX_Credit_Max_Overdu_Amount', 'ZX_Credit_Max_Overdue_Duration']

# Remover colunas do DataFrame
df_select = df.drop(del_cols, axis=1)

# Visualizar primeiras linhas do novo conjunto de dados
df_select.head()

# %%

# 4.2 CONSTRUÇÃO DE ATRIBUTOS POLINOMIAIS

from sklearn.preprocessing import PolynomialFeatures

# Selecionar variáveis para construção polinomial
poly_feature = df[['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']].fillna(0)

# Definir transformação polinomial
poly_trans = PolynomialFeatures(degree=3)

# Ajustar modelo
ptf = poly_trans.fit(poly_feature)

# Transformar dados
poly_feature = poly_trans.transform(poly_feature)

print("Shape após transformação polinomial:", poly_feature.shape)

# %%

# 4.2.2 ANÁLISE DAS NOVAS VARIÁVEIS

# Obter nomes das novas features (compatível com versões novas e antigas do sklearn)
try:
    feature_names = poly_trans.get_feature_names_out(
        ['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']
    )
except AttributeError:
    feature_names = poly_trans.get_feature_names(
        ['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']
    )

# Criar DataFrame com novas features
poly_features = pd.DataFrame(poly_feature, columns=feature_names, index=df.index)

# Adicionar variável alvo
poly_features['Target'] = df['Target']

# Visualizar dados
poly_features.head()

# %%

# 4.2.3 ANÁLISE DE CORRELAÇÃO COM A VARIÁVEL ALVO

# Calcular correlação com variável alvo
poly_corrs = poly_features.corr()['Target'].sort_values()

# Exibir variáveis com menor correlação
print("Cinco atributos com os menores coeficientes de correlação:\n",
      poly_corrs.head(5))

# Exibir variáveis com maior correlação
print("\nCinco atributos com os maiores coeficientes de correlação:\n",
      poly_corrs.tail(5))

# %%
