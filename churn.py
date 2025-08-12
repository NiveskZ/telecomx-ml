#%%
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
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
    (OneHotEncoder(), cat_cols),
    remainder='passthrough',
    sparse_threshold=0
    )

# %%
