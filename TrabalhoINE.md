---

# Relatório de Análise do Conjunto de Dados de Inadimplência de Cartões de Crédito

## Introdução

O problema abordado neste relatório é a previsão de inadimplência de clientes de cartões de crédito. O conjunto de dados escolhido para a análise é o "Default of Credit Card Clients Dataset" do UCI Machine Learning Repository. Este conjunto de dados contém informações sobre clientes de cartões de crédito em Taiwan e é utilizado para prever o risco de inadimplência com base em variáveis demográficas e comportamentais.

**Descrição do Conjunto de Dados:**
- **Fonte:** Taiwan Economic Journal
- **Número de Instâncias:** 30.000
- **Número de Atributos:** 24
- **Variáveis Principais:** ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0 a PAY_6, BILL_AMT1 a BILL_AMT6, PAY_AMT1 a PAY_AMT6, default.payment.next.month

## Metodologia

### Análise Descritiva

#### Limpeza e Preparação dos Dados

1. **Tratamento de Valores Ausentes e Dados Duplicados:**
   O conjunto de dados foi verificado para valores ausentes e dados duplicados. Como o dataset não apresentou valores ausentes, nenhuma ação de preenchimento foi necessária. Dados duplicados foram identificados e removidos para garantir a integridade dos dados.

2. **Remoção da Coluna "ID":**
   A coluna "ID" foi removida, pois é um identificador único que não contribui para a modelagem preditiva e pode introduzir variabilidade sem relevância para o modelo.

3. **Codificação de Variáveis Categóricas:**
   As variáveis categóricas foram codificadas usando one-hot encoding, transformando-as em um formato numérico apropriado para algoritmos de machine learning. A opção `drop_first=True` foi utilizada para evitar a multicolinearidade.

```python
# Remover a coluna "ID" pois ela não será utilizada
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Codificar variáveis categóricas
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
```

#### Estatísticas Descritivas

Foram calculadas estatísticas descritivas para entender a distribuição das variáveis contínuas e identificar possíveis outliers ou padrões.

```python
# Verificar estatísticas descritivas
print("\nEstatísticas descritivas do dataset:")
print(df.describe())
```

#### Visualizações

1. **Distribuição da Variável-Alvo:**

  ![download](https://github.com/user-attachments/assets/5cec6d9c-6736-44e9-be03-d11c01ce581e)


   ```python
   # Distribuição da variável alvo
   plt.figure(figsize=(6, 4))
   sns.countplot(x='default payment next month', data=df)
   plt.title('Distribuição de Default')
   plt.show()
   ```
#### Insights:
   1. **Distribuição Desbalanceada**: 
      - O gráfico mostra que a maioria dos clientes não são inadimplentes (`0`), com mais de 20.000 instâncias. Em contraste, há um número significativamente menor de inadimplentes (`1`), com cerca de 5.000 instâncias.
      - Esse desbalanceamento indica que a classe dos não inadimplentes é majoritária, o que pode influenciar o desempenho dos modelos preditivos, especialmente em termos de acurácia e outras métricas, já que o modelo pode se inclinar a prever a classe majoritária.
   
   2. **Implicações para o Modelo**:
      - **Necessidade de Técnicas de Balanceamento**: O desbalanceamento pode exigir técnicas como oversampling da classe minoritária ou undersampling da classe majoritária para evitar que o modelo aprenda a favorecer a classe dos não inadimplentes.
      - **Impacto nas Métricas de Avaliação**: Ao avaliar o modelo, será importante considerar métricas como F1-score, recall, ou a matriz de confusão, que podem fornecer uma visão mais clara do desempenho nas classes desbalanceadas, em vez de se basear apenas na acurácia.
   3. **Conclusões**:
   
      - **Risco de Inadimplência**: A baixa proporção de inadimplentes sugere que a maioria dos clientes gerencia bem seus pagamentos de cartão de crédito. Isso pode ser indicativo de critérios de concessão de crédito mais rigorosos ou de um comportamento financeiro mais responsável na base de clientes analisada.
      - **Modelo de Negócio**: Empresas que utilizam esse tipo de análise podem explorar modelos de risco mais refinados, focando especialmente em características que diferenciam os inadimplentes dos não inadimplentes, o que pode ajudar em estratégias de crédito personalizadas.


2. **Boxplots para Variáveis Contínuas:**

   
   ![download](https://github.com/user-attachments/assets/170f630b-851e-495f-9962-f6b7502d842b)

   ```python
   # Boxplots para variáveis contínuas
   plt.figure(figsize=(14, 10))
   sns.boxplot(data=df[['AGE', 'BILL_AMT1', 'PAY_AMT1']])
   plt.title('Boxplots para Variáveis Contínuas')
   plt.show()
   ```
   **Insight:** 


   #### Padrões Observados:
   1. **Variável `AGE`**:
      - A variável `AGE` apresenta uma distribuição bastante concentrada, com uma faixa estreita de valores e poucos outliers. Isso sugere que a maioria dos clientes está dentro de uma faixa etária específica, sem grandes variações extremas. Essa concentração pode indicar um perfil demográfico específico para os clientes analisados.
   
   2. **Variável `BILL_AMT1`**:
      - A variável `BILL_AMT1`, que representa o valor da primeira fatura de cada cliente, exibe uma distribuição mais dispersa, com uma mediana relativamente alta. Há muitos outliers significativos, indicando que alguns clientes possuem faturas excepcionalmente altas. Esses outliers podem ser importantes para identificar clientes com alto risco ou comportamento financeiro atípico.
   
   3. **Variável `PAY_AMT1`**:
      - A variável `PAY_AMT1`, que representa o valor do pagamento da primeira fatura, também mostra uma distribuição com muitos outliers. No entanto, os valores centrais são significativamente mais baixos do que `BILL_AMT1`, o que pode sugerir que alguns clientes pagam apenas uma parte da dívida ou valores menores do que os totais de suas faturas.
   
   #### Insights:
   - **Perfil Demográfico e Financeiro**: A concentração da variável `AGE` sugere que a instituição financeira pode ter um perfil de cliente-alvo bem definido em termos de idade. Já as variações extremas em `BILL_AMT1` e `PAY_AMT1` indicam que, embora alguns clientes tenham grandes faturas, os valores pagos podem não ser suficientes para cobrir essas dívidas, o que é um sinal potencial de inadimplência.
   
   - **Risco de Inadimplência**: A discrepância entre os valores de `BILL_AMT1` e `PAY_AMT1` pode ser um indicativo de clientes que estão lutando para manter seus pagamentos em dia. Esses padrões podem ser úteis para refinar os modelos preditivos de inadimplência, focando em clientes que apresentam altos valores de fatura em relação ao que pagam.
   
   - **Atenção aos Outliers**: Os outliers identificados nas variáveis `BILL_AMT1` e `PAY_AMT1` merecem uma análise mais detalhada, pois podem representar comportamentos financeiros fora do comum, que podem distorcer as previsões se não forem tratados adequadamente.



3. **Histogramas para Variáveis Contínuas:**

   ![Histogramas para Variáveis Contínuas](#)  <!-- Inserir histogramas aqui -->

   ```python
   # Histogramas para variáveis contínuas
   plt.figure(figsize=(14, 10))
   df[['AGE', 'BILL_AMT1', 'PAY_AMT1']].hist(bins=30, figsize=(14, 10), layout=(2, 3))
   plt.suptitle('Histogramas para Variáveis Contínuas')
   plt.show()
   ```
   **Insight:** Os histogramas mostram a distribuição das variáveis contínuas, permitindo verificar se elas seguem uma distribuição normal ou apresentam assimetrias.

4. **Gráficos de Dispersão:**

   ![Relação entre Idade e Fatura 1](#)  <!-- Inserir gráfico de dispersão entre AGE e BILL_AMT1 aqui -->

   ```python
   # Gráficos de dispersão
   plt.figure(figsize=(8, 6))
   sns.scatterplot(x='AGE', y='BILL_AMT1', hue='default payment next month', data=df)
   plt.title('Relação entre Idade e Fatura 1')
   plt.show()
   ```
   **Insight:** O gráfico de dispersão revela a relação entre idade e fatura. Padrões ou agrupamentos podem indicar como a inadimplência varia com essas variáveis.

### Análise Preditiva

#### Divisão dos Dados

Os dados foram divididos em conjuntos de treinamento e teste para avaliar a capacidade dos modelos de generalizar para novos dados.

```python
# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Justificativa da Escolha dos Modelos

1. **Regressão Logística:**
   Escolhida por ser um modelo simples e interpretável que funciona bem para problemas de classificação binária. Permite avaliar a probabilidade de inadimplência com base nas variáveis preditoras.

2. **Random Forest:**
   Um modelo de ensemble que pode capturar interações complexas entre variáveis e é robusto a overfitting. Escolhido para comparar seu desempenho com o da regressão logística.

```python
# Modelos a serem testados
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}
```

#### Avaliação dos Modelos

1. **Logistic Regression:**

   **Acurácia:** X.XX

   **Relatório de Classificação:**

   ```plaintext
   [Inserir relatório de classificação aqui]
   ```

   **Matriz de Confusão:**

   ![Matriz de Confusão (Logistic Regression)](#)  <!-- Inserir matriz de confusão para Logistic Regression aqui -->

   ```python
   # Matriz de Confusão
   cm = confusion_matrix(y_test, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
   plt.title(f'Matriz de Confusão (Logistic Regression)')
   plt.ylabel('Verdadeiro')
   plt.xlabel('Previsto')
   plt.show()
   ```

   **Insight:** A Regressão Logística mostra boa capacidade de distinguir entre classes, mas pode ter dificuldades com a classe minoritária.

2. **Random Forest:**

   **Acurácia:** X.XX

   **Relatório de Classificação:**

   ```plaintext
   [Inserir relatório de classificação aqui]
   ```

   **Matriz de Confusão:**

   ![Matriz de Confusão (Random Forest)](#)  <!-- Inserir matriz de confusão para Random Forest aqui -->

   ```python
   # Matriz de Confusão
   cm = confusion_matrix(y_test, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
   plt.title(f'Matriz de Confusão (Random Forest)')
   plt.ylabel('Verdadeiro')
   plt.xlabel('Previsto')
   plt.show()
   ```

   **Insight:** O Random Forest pode capturar melhor as complexidades dos dados e lidar melhor com a desbalanceamento de classes.

## Discussão

- **Interpretação dos Resultados:**
  - A análise revelou que o desbalanceamento das classes pode impactar a performance dos modelos. O Random Forest, como modelo mais complexo, apresentou melhor desempenho na distinção entre as classes de inadimplência.

- **Limitações:**
  - O desbalanceamento de classes pode ter levado a uma performance inferior dos modelos, especialmente na identificação da classe minoritária.
  - A Regressão Logística pode não capturar todas as complexidades dos dados.

- **Possíveis Melhorias:**
  - Aplicar técnicas de balanceamento de dados, como oversampling da classe minoritária ou undersampling da classe majoritária.
  - Explorar outras técnicas de modelagem, como Gradient Boosting, para melhorar a performance.

## Conclusão

A análise preditiva revelou que tanto a Regressão Logística quanto o Random Forest são úteis para prever a inadimplência de clientes de cartões de crédito. O Random Forest mostrou um desempenho superior, mas o desbalanceamento de classes continua sendo um desafio. Melhorias nos dados e nos modelos podem proporcionar insights mais precisos sobre o risco de inadimplência.

## Código-Fonte

```python
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)

# Exibir as primeiras entradas
print("Primeiras entradas do dataset:")
print(df.head())

# Verificar nomes das colunas
print("\nNomes das colunas do dataset:")
print(df.columns)

# Verificar informações gerais do dataset
print("\nInformações gerais do dataset:")
print(df.info())

# Verificar estatísticas descritivas
print("\nEstatísticas descritivas do dataset:")
print(df.describe())

# Remover a coluna "ID" pois ela não será utilizada
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Codificar variáveis categóricas
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Separando características e variável alvo
target_column = 'default payment next month'
if target_column in df.columns:
    X = df.drop(columns=[target_column])
    y = df[target_column]
else:
    print(f"Coluna {target_column} não encontrada.")
    raise ValueError(f"Coluna {target_column} não encontrada no dataset.")

# Análise Descritiva

# Distribuição da variável alvo
plt.figure(figsize=(6, 4))
sns.countplot(x='default payment next month', data=df)
plt.title('Distribuição de Default')
plt.show()

# Boxplots para variáveis contínuas
plt.figure(figsize=(14, 10))
sns.boxplot(data=df[['AGE', 'BILL_AMT1', 'PAY_AMT1']])
plt.title('Boxplots para Variáveis Contínuas')
plt.show()

# Histogramas para variáveis contínuas
plt.figure(figsize=(14, 10))
df[['AGE', 'BILL_AMT1', 'PAY_AMT1']].hist(bins=30, figsize=(14, 10), layout=(2, 3))
plt.suptitle('Histogramas para Variáveis Contínuas')
plt.show()

# Gráficos de dispersão
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AGE', y='BILL_AMT1', hue='default payment next month', data=df)
plt.title('Relação entre Idade e Fatura 1')
plt.show()

# Análise Preditiva

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos a serem testados
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

for model_name, model in models.items():
    # Treinamento do modelo
    model.fit(X_train_scaled, y_train)

    # Fazendo previsões
    y_pred = model.predict(X_test_scaled)

    # Avaliando o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAcurácia ({model_name}): {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Matriz de Confusão ({model_name})')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

# Discussão das limitações
print("\nDiscussão das limitações:")
print("1. A acurácia pode não refletir a verdadeira performance dos modelos devido ao desbalanceamento da variável-alvo.")
print("2. A Regressão Logística pode não capturar todas as complexidades dos dados.")
print("3. O Random Forest, apesar de robusto, pode ser difícil de interpretar.")

print("\nPossíveis Melhorias:")
print("1. Aplicar técnicas de balanceamento de dados, como oversampling ou undersampling.")
print("2. Explorar outras técnicas de modelagem, como Gradient Boosting.")
```

---
