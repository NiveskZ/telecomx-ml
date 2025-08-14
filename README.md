# Telecom X – Parte 2: Prevendo Churn
![Static Badge](https://img.shields.io/badge/status-em_desenvolvimento-blue)

Desafio promovido pela Oracle Next Education junto da Alura.

## Ferramentas utilizadas
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- statsmodels
- yellowbricks

## Objetivos
Meu desafio aqui é é desenvolver modelos preditivos capazes de prever quais clientes têm maior chance de cancelar seus serviços devido a demanda da empresa de antecipar o problema de evasão.

O que vai ser realizado:
- Preparar os dados para a modelagem (tratamento, encoding, normalização).

- Realizar análise de correlação e seleção de variáveis.

- Treinar dois ou mais modelos de classificação.

- Avaliar o desempenho dos modelos com métricas.

- Interpretar os resultados, incluindo a importância das variáveis.

- Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.

## Preparação de Dados
Como nossa fonte de dados já tinha sido tratado, em sua maioria na parte de ETL, aqui só precisamos fazer algumas modificações básicas para evitar data leakage e multicolinearidade para modelos lineares:

- Foi removido a coluna de identificação `customerID`

- Colunas que possuiam valores `No internet service` foi reatribuido o valor `No`, uma vez que para o modelo preditivio significa a mesma coisa e geraria multicolinearidade para modelos lineares.

- Para transformar colunas categóricas em numéricas foi utilizado a codificação One-Hot através do `OneHotEncoder` da biblioteca `scikit-learn` 

- Normalização de valores numéricos para garantir eficiência de modelos lineares e baseados em distância.

Feita essas mudanças, conseguimos comparar entre diferentes tipos de modelos qual nos trás a melhor resposta para nosso problema.

## Análise de Correlação e Seleção de Variáveis

Nessa etapa foi criada matrizes de correlação para reduzir multicolinearidades e entender quais variáveis respondem melhor o que faz o cliente dar ou não churn. Inicialmente foi criado uma matriz de correlação geral com todas as variáveis, obtendo a imagem abaixo contendo apenas a linha em relação ao `Churn`.

![Churn](imgs\churn_corr.png)

Com objetivo de selecionar apenas variáveis minimamente relevantes, foi decidido pegar um limiar de valor maior que 0.15 (valor absoluto). Podemos ter uma visualização ampliada dessas variáveis através do gráfico a seguir.

![Heatmap das variáveis com correlação >= 0.15 com Churn](imgs\corr_015.png)

Note que várias variáveis apresentam alta colinearidade entre si, o que pode prejudicar modelos lineares. Para lidar com isso, utilizamos a biblioteca `statsmodels`.

Primeiro, aplicamos o Fator de Inflação da Variância (VIF) para detectar multicolinearidade. Variáveis com VIF muito alto indicam que estão fortemente correlacionadas com outras variáveis explicativas, o que pode inflar os coeficientes e tornar o modelo instável. Removemos essas variáveis prioritariamente, mantendo apenas as mais representativas.

Em seguida, ajustamos um modelo de Mínimos Quadrados Ordinários (OLS) para obter um sumário estatístico completo, incluindo coeficientes, erros padrão, intervalos de confiança e métricas de significância. Com base nesses resultados, selecionamos as variáveis mais relevantes utilizando o teste t e o p-valor.

O critério adotado foi `p-valor < 0.05`, ou seja, apenas variáveis cujo efeito é estatisticamente significativo ao nível de 95% foram mantidas. Esse processo nos permite identificar quais variáveis realmente influenciam a variável resposta (`Churn`) e garantir maior confiabilidade e interpretabilidade para modelos lineares subsequentes.

## Modelagem Preditiva
Para definir um ponto de partida foi utilizado `DummyClassifier` que da uma acurácia inicial de `0.73` a ser batido. Além dessa métrica também foi utilizada o ROC AUC para definir o melhor modelo.

Foram utilizado três principais modelos como base:
- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier` 

Desses o único que teve um tuning de hiperparâmetros foi o `RandomForestClassifier` através do `GridSearchCV` para facilitar a escolha dos melhores parâmetros.

Obtendo um AUC de `0.84` foi escolhido como modelo o `RandomForestClassifier`, a partir dele podemos olhar para os gráficos abaixo que nos mostram a importância de cada variável:

![Feature Importance](imgs\feature_importance.png)

## Conclusão

Observamos então que as cobranças totais e os contratos mensais são os principais responsáveis por Churn do cliente na empresa, além dos clientes que possuem internet com fibra ótica, podendo estar ligado com a qualidade do produto. Também observamos que clientes que estão a um longo tempo dentro da empresa tem um impacto positivo, evitando o Churn.

