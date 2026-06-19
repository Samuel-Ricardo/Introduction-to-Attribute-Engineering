
#%%

import pandas as pd


df=pd.read_csv("../[ES_510]_Lab01/credit.csv");

df.head()


import missingno as msno


msno.matrix(df)


df_missing = pd.DataFrame(df.isnull().sum()/df.shape[0], columns=['missing_rate']).reset_index()

df_missing.sort_values(by='missing_rate', ascending=False)[:15]


missing_col = ['Title', 'Industry', 'House_State', 'Nation', 'Marriage_State', 'Highest Education', 'Duty']

df_missing_2 = pd.DataFrame(df.isnull().sum()/df.shape[0], columns=['missing_rate']).reset_index()
df_missing_2.sort_values(by='missing_rate', ascending=False)[:15]


cross_table = pd.crosstab(df.House_State, columns = df.Target, margins = True)

cross_table_rowpct = cross_table.div(cross_table['All'], axis = 0)
cross_table_rowpct


for col in missing_col:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop('Target', axis = 1)
y = df['Target']

X_category = df[['Nation', 'Birth_Place', 'Gender', 'Marriage_State', 'Highest Education', 'House_State', 'Work_Years', 'Title', 'Duty', 'Industry']]


from sklearn.feature_selection import chi2

chi2_test = chi2


(chi2_vals, pval) = chi2_test(X_category, y)


dict_feature = {}


for i, j in zip(X_category.columns.values, chi2_vals):
    dict_feature[i] = j


ls = sorted(dict_feature.items(), key=lambda item: item[1], reverse=True)


ls


import matplotlib.pyplot as plt
import seaborn as sns


nominal_features = ['Nation', 'Birth_Place', 'Gender', 'Marriage_State',
                    'Highest Education', 'House_State', 'Work_Years',
                    'Unit_Kind', 'Title', 'Occupation', 'Duty', 'Industry']


numerical_features = [col for col in X.columns if col not in nominal_features]


numerical_features.pop(0)


X_num = df[numerical_features]


corr_matrix = X_num.corr(method='spearman')


plt.figure(figsize=(16, 12))


sns.heatmap(corr_matrix, annot=True, fmt='.2g', cmap='magma')
plt.tight_layout()
plt.show()


cols_pair = []
for index_ in corr_matrix.index:
    for col_ in corr_matrix.columns:
        if corr_matrix.loc[index_, col_] >= 0.8 and index_ != col_ and (col_, index_) not in cols_pair:
            cols_pair.append((index_, col_))


cols_pair


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


X = df.drop('Target', axis=1).fillna(0)
y = df['Target']


model = LogisticRegression(max_iter=1000, solver='liblinear')


x_rfe = RFE(estimator=model, n_features_to_select=10).fit(X, y)


print("Número de atributos selecionados:", x_rfe.n_features_)


print("Suporte (máscara True/False):", x_rfe.support_)


print("Ranking dos atributos:", x_rfe.ranking_)


print("Modelo final:", x_rfe.estimator_)


selected_features = X.columns[x_rfe.support_].tolist()
print("Atributos selecionados pelo RFE:", selected_features)


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=100, random_state=42)


rfc.fit(X, y)


cols = X.columns.tolist()


sorted_feature = sorted(
    zip(map(lambda x: round(x, 4), rfc.feature_importances_), cols),
    reverse=True
)


sorted_feature


del_cols = ['Gender', 'House_State', 'Couple_Year_Income', 'Loan_Curr_Bal',
            'ZX_Max_Credit_Banks', 'ZX_Max_Overdue_Credits',
            'ZX_Credit_Max_Overdu_Amount', 'ZX_Credit_Max_Overdue_Duration']


df_select = df.drop(del_cols, axis=1)


df_select.head()


from sklearn.preprocessing import PolynomialFeatures


poly_feature = df[['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']].fillna(0)


poly_trans = PolynomialFeatures(degree=3)


ptf = poly_trans.fit(poly_feature)


poly_feature = poly_trans.transform(poly_feature)

print("Shape após transformação polinomial:", poly_feature.shape)


try:
    feature_names = poly_trans.get_feature_names_out(
        ['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']
    )
except AttributeError:
    feature_names = poly_trans.get_feature_names(
        ['Ast_Curr_Bal', 'Age', 'Year_Income', 'Std_Cred_Limit']
    )


poly_features = pd.DataFrame(poly_feature, columns=feature_names, index=df.index)


poly_features['Target'] = df['Target']


poly_features.head()


poly_corrs = poly_features.corr()['Target'].sort_values()


print("Cinco atributos com os menores coeficientes de correlação:\n",
      poly_corrs.head(5))


print("\nCinco atributos com os maiores coeficientes de correlação:\n",
      poly_corrs.tail(5))


