# Telecom X – Parte 2: Prevendo Churn
![Static Badge](https://img.shields.io/badge/status-em_desenvolvimento-blue)

Desafio promovido pela Oracle Next Education junto da Alura.

## Ferramentas utilizadas
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

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