#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#%%
url = 'https://raw.githubusercontent.com/NiveskZ/telecomx-etl/refs/heads/main/data/telecomx.csv'
df = pd.read_csv(url)
df.head()
# %%
df = df.drop(columns=['customerID'])
# %%
df.info()
# %%
df_fix = df.copy()
# Colunas com elementos de categoria 'no internet service'
cols_to_fix = ['OnlineSecurity','OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies']

for col in cols_to_fix:
    df_fix[col] = df_fix[col].replace('No internet service','No')

df_fix.head()
# %%
# Enconding
cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaymentMethod']

one_hot = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), cat_cols),
    remainder='passthrough',
    sparse_threshold=0
    )

# %%
# Normalizando os dados para modelos baseados em distância
num_cols = df.select_dtypes(include=['int64','float64']).columns

scaler = MinMaxScaler()
df_fix[num_cols] = scaler.fit_transform(df_fix[num_cols])

df.head()
# %%
print("Tem NaN?", df_fix.isna().any().any())
print("Tem Inf?", np.isinf(df_fix[num_cols]).any().any())

# Mostrar colunas problemáticas
nan_cols = df_fix.columns[df_fix.isna().any()]
inf_cols = num_cols[np.isinf(df_fix[num_cols]).any()]
print("Colunas com NaN:", nan_cols)
print("Colunas com Inf:", inf_cols)
# %%
df_fix['Charges_Total_Day'].isna().sum()

#%%
df_fix['Charges_Total_Day'].fillna(0, inplace=True)
# %%
y = df_fix['Churn']
X = df_fix.drop('Churn', axis=1)
cols = X.columns
# %%
X_encoded = one_hot.fit_transform(X)
cols = one_hot.get_feature_names_out(cols)

X_encoded = pd.DataFrame(X_encoded, columns=cols)
# %%
# Análise de correlação
df_corr = pd.DataFrame(X_encoded, columns=cols)
df_corr['Churn'] = y 

# %%
# Visualização heatmap
fig, ax = plt.subplots(figsize=(20,16))
ax = sns.heatmap(np.round(df_corr.corr(), 2), vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
plt.show()
# %%
threshold = 0.15
corr = df_corr.corr()

# Seleciona variáveis mais relevantes baseado no nosso limiar de 0.15
var_selection = corr.index[abs(corr['Churn']) >= threshold].to_list()
# Faz uma nova matriz de correlação apenas com essas novas variáveis
corr_selection = corr.loc[var_selection,var_selection]

# Gerar uma máscara para esconder o triângulo superior da matriz (incluindo diagonal)
mascara = np.triu(np.ones_like(corr_selection, dtype=bool))

# Plotar o heatmap com a máscara aplicada para melhor visualização
plt.figure(figsize=(12,10))
sns.heatmap(
    corr_selection,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    mask=mascara
)
plt.title(f'Heatmap das variáveis com correlação >= {threshold} com Churn')
plt.show()
# %%
features = [col for col in var_selection if col != 'Churn']
X_encoded = X_encoded[features]

print(y.value_counts(normalize=True))
X_encoded.head()
# %%
vif = pd.DataFrame()
vif["feature"] = X_encoded.columns
vif['VIF'] = [variance_inflation_factor(X_encoded.values, i) 
              for i in range(X_encoded.shape[1])]
print(vif.sort_values("VIF",ascending=False))
# %%
X_encoded.drop(['remainder__Charges_Monthly', 'remainder__Charges_Day',
                'remainder__Charges_Total_Day'], axis=1, inplace=True)
# %%

vif = pd.DataFrame()
vif["feature"] = X_encoded.columns
vif['VIF'] = [variance_inflation_factor(X_encoded.values, i) 
              for i in range(X_encoded.shape[1])]
print(vif.sort_values("VIF",ascending=False))
# %%
X, X_test, y, y_test = train_test_split(X_encoded,y, test_size=.15,
                                     stratify=y, random_state=42)
# %%
X_train, X_val, y_train, y_val = train_test_split(X,y,
                                                stratify=y, random_state=42)

# %% 
X_train_const = sm.add_constant(X_train)
ols = sm.OLS(y_train,X_train_const, hasconst=True).fit()
print(ols.summary())
# %%
X_train_const.drop(['onehotencoder__Contract_One year',
                    'remainder__Partner','remainder__Dependents'],axis=1,inplace=True)
# %%
ols = sm.OLS(y_train,X_train_const, hasconst=True).fit()
print(ols.summary())
# %%
features = X_train_const.drop('const',axis=1).columns
# %%
X_train = X_train[features]
X_val = X_val[features]
X_test = X_test[features]
# %%
dummy = DummyClassifier()
dummy.fit(X_train,y_train)

dummy.score(X_val,y_val)
# %%

linear_model = LinearRegression()
logistic_model = LogisticRegression(random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)