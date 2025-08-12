#%%
import pandas as pd

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
# %%
y = df_fix['Churn']
X = df_fix.drop('Churn', axis=1)
cols = X.columns
# %%
X = one_hot.fit_transform(X)
cols = one_hot.get_feature_names_out(cols)
# %%
X = pd.DataFrame(X,columns=cols )
print(y.value_counts(normalize=True))
X.head()
# %%
X, X_test, y, y_test = train_test_split(X,y, test_size=.15,
                                     stratify=y, random_state=42)
# %%
X_train, X_val, y_train, y_val = train_test_split(X,y,
                                                stratify=y, random_state=42)
#%% 
dummy = DummyClassifier()

# %%
linear_model = LinearRegression()
logistic_model = LogisticRegression(random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)