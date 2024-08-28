---


# Relatório de Análise do Conjunto de Dados de Inadimplência de Cartões de Crédito
   ### **Autores:** Maressa Karen e Mateus Lannes

## Introdução

O problema abordado neste relatório é a previsão de inadimplência de clientes de cartões de crédito. O conjunto de dados escolhido para a análise é o "Default of Credit Card Clients Dataset" do UCI Machine Learning Repository. Este conjunto de dados contém informações sobre clientes de cartões de crédito em Taiwan e é utilizado para prever o risco de inadimplência com base em variáveis demográficas e comportamentais.

### **Descrição do Conjunto de Dados:**      

O dataset "Default of Credit Card Clients" do UCI Machine Learning Repository contém informações detalhadas sobre clientes de cartões de crédito em Taiwan, com o objetivo de prever o risco de inadimplência no próximo mês. Aqui estão as informações completas sobre este dataset:

### Informações Gerais:
- **Título**: Default of Credit Card Clients Dataset
- **Fonte**: Taiwan Economic Journal
- **Número de Instâncias (Registros)**: 30.000
- **Número de Atributos (Colunas)**: 24

### Descrição dos Atributos:
1. **ID**: Identificador único do cliente.
2. **LIMIT_BAL**: Valor do crédito concedido ao cliente (em dólares NT).
3. **SEX**: Gênero do cliente:
   - 1 = Masculino
   - 2 = Feminino
4. **EDUCATION**: Nível de escolaridade do cliente:
   - 1 = Pós-graduação
   - 2 = Graduação
   - 3 = Ensino médio
   - 4 = Outros
   - 5 = Desconhecido
   - 6 = Desconhecido
5. **MARRIAGE**: Estado civil do cliente:
   - 1 = Casado
   - 2 = Solteiro
   - 3 = Outros
6. **AGE**: Idade do cliente (em anos).
7. **PAY_0**: Histórico de pagamento em setembro de 2005:
   - 0 = Pagamento dentro do prazo
   - 1-9 = Atrasos de 1 a 9 meses
8. **PAY_2**: Histórico de pagamento em agosto de 2005.
9. **PAY_3**: Histórico de pagamento em julho de 2005.
10. **PAY_4**: Histórico de pagamento em junho de 2005.
11. **PAY_5**: Histórico de pagamento em maio de 2005.
12. **PAY_6**: Histórico de pagamento em abril de 2005.
13. **BILL_AMT1**: Valor da fatura em setembro de 2005.
14. **BILL_AMT2**: Valor da fatura em agosto de 2005.
15. **BILL_AMT3**: Valor da fatura em julho de 2005.
16. **BILL_AMT4**: Valor da fatura em junho de 2005.
17. **BILL_AMT5**: Valor da fatura em maio de 2005.
18. **BILL_AMT6**: Valor da fatura em abril de 2005.
19. **PAY_AMT1**: Valor pago em setembro de 2005.
20. **PAY_AMT2**: Valor pago em agosto de 2005.
21. **PAY_AMT3**: Valor pago em julho de 2005.
22. **PAY_AMT4**: Valor pago em junho de 2005.
23. **PAY_AMT5**: Valor pago em maio de 2005.
24. **PAY_AMT6**: Valor pago em abril de 2005.
25. **default.payment.next.month**: Indicador de inadimplência do próximo mês:
    - 1 = Inadimplente
    - 0 = Não inadimplente


### Descrição:
Este dataset é frequentemente utilizado para prever a inadimplência de clientes de cartões de crédito com base em uma variedade de atributos demográficos e financeiros. A variável-alvo é `default.payment.next.month`, que indica se o cliente será ou não inadimplente no mês seguinte.

### Aplicações:
Este dataset é amplamente usado em análises de risco de crédito e em modelos de aprendizado de máquina para prever a inadimplência. É um recurso valioso para estudos em finanças, ciência de dados e inteligência artificial.

Este conjunto de dados pode ser utilizado para treinar modelos de classificação e análise preditiva, auxiliando instituições financeiras na gestão de risco e na tomada de decisões.



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

  ![Distribuição da Variável-Alvo](https://github.com/user-attachments/assets/5cec6d9c-6736-44e9-be03-d11c01ce581e)


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

   
   ![Boxplots para Variáveis Contínuas](https://github.com/user-attachments/assets/170f630b-851e-495f-9962-f6b7502d842b)

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

    ![Histogramas para Variáveis Contínuas](https://github.com/user-attachments/assets/f762e745-dba4-4b25-8c0f-659a1d94f156)


   ```python
   # Histogramas para variáveis contínuas
   plt.figure(figsize=(14, 10))
   df[['AGE', 'BILL_AMT1', 'PAY_AMT1']].hist(bins=30, figsize=(14, 10), layout=(2, 3))
   plt.suptitle('Histogramas para Variáveis Contínuas')
   plt.show()
   ```
   **Insight:**
      ### 1. **Histograma da Idade (`AGE`)**
      - **Distribuição:** A distribuição da idade é assimétrica à direita, com a maior concentração de clientes na faixa de 25 a 40 anos. Poucos clientes têm mais de 60 anos, e quase nenhum tem menos de 20 ou mais de 70 anos.
      - **Insight:** A maioria dos clientes de cartão de crédito nesta amostra são adultos jovens a meia-idade. Isso pode indicar que as pessoas nessa faixa etária são mais propensas a utilizar crédito. Este grupo pode ser mais relevante para estratégias de marketing e gestão de risco de crédito.
   
   ### 2. **Histograma do Valor da Fatura de Setembro de 2005 (`BILL_AMT1`)**
      - **Distribuição:** A distribuição é altamente concentrada em valores próximos de zero, com uma cauda longa à direita. Isso indica que a maioria das faturas tem valores baixos, mas há alguns casos de valores de fatura extremamente altos (até 1 milhão de dólares NT).
      - **Insight:** Embora a maioria dos clientes mantenha saldos relativamente baixos em suas faturas, a existência de alguns valores muito altos pode indicar a presença de um pequeno grupo de clientes com grande capacidade de crédito ou grandes dívidas. Esses clientes podem representar tanto um grande potencial de lucro quanto um risco elevado de inadimplência.
   
   ### 3. **Histograma do Valor Pago em Setembro de 2005 (`PAY_AMT1`)**
      - **Distribuição:** A maioria dos pagamentos feitos pelos clientes também está concentrada em valores muito baixos, com alguns casos isolados de pagamentos muito altos (acima de 200 mil dólares NT).
      - **Insight:** A concentração de pagamentos baixos pode refletir a estratégia de pagamento mínima por parte dos clientes, onde muitos preferem pagar o valor mínimo devido, resultando em altas taxas de juros sobre o saldo restante. Os poucos pagamentos altos podem representar liquidações de dívidas substanciais ou clientes que preferem manter seus saldos baixos.
   
   ### **Considerações Finais:**
   - **Desigualdade Financeira:** A presença de caudas longas à direita nos histogramas de `BILL_AMT1` e `PAY_AMT1` sugere uma desigualdade no comportamento financeiro dos clientes, com uma minoria exibindo saldos e pagamentos muito maiores do que a maioria.
   - **Segmentação de Mercado:** A análise sugere que a segmentação dos clientes em diferentes faixas etárias e comportamentais pode ser útil para a personalização de ofertas e estratégias de mitigação de risco.
   - **Risco e Potencial de Lucro:** Os poucos clientes com saldos e pagamentos muito elevados podem representar tanto um alto potencial de lucro quanto um risco de inadimplência, exigindo uma análise mais detalhada e um acompanhamento rigoroso.





5. **Gráficos de Dispersão:**

     ![Gráficos de Dispersão](https://github.com/user-attachments/assets/81e562a9-805a-4afc-b61b-de0a54f62eb0)


   ```python
   # Gráficos de dispersão
   plt.figure(figsize=(8, 6))
   sns.scatterplot(x='AGE', y='BILL_AMT1', hue='default payment next month', data=df)
   plt.title('Relação entre Idade e Fatura 1')
   plt.show()
   ```
   **Insight:**
      1. **Distribuição Etária e Valores de Fatura**
         - **Concentração Etária:** A maior parte dos clientes tem entre 25 e 60 anos, com uma leve concentração entre 30 e 50 anos. Este é o grupo que parece ser o mais ativo em termos de uso de crédito.
      
      2. **Inadimplência em Relação à Idade**
         - **Distribuição de Inadimplentes:** Clientes inadimplentes estão presentes em todas as faixas etárias, mas não há uma concentração muito alta em faixas etárias específicas. Isso sugere que a idade por si só não é um fator isolado determinante para inadimplência.
      
      3. **Inadimplência em Relação ao Valor da Fatura**
         - **Altos Valores de Fatura:** Acima de 500 mil dólares NT, a quantidade de inadimplentes diminui. Isso pode indicar que clientes com faturas mais altas estão mais bem posicionados financeiramente ou que têm acesso a recursos que evitam a inadimplência.
      
      4. **Padrões Gerais de Inadimplência**
         - **Ausência de Padrões Claros:** A ausência de uma tendência clara entre idade, valor da fatura e inadimplência sugere que o comportamento de inadimplência é complexo e possivelmente influenciado por uma combinação de fatores demográficos e comportamentais (como histórico de pagamento, renda, nível educacional, etc.).







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

   **Acurácia:** 0.81

   **Relatório de Classificação:**

   ```plaintext

              precision    recall  f1-score   support
           0       0.82      0.97      0.89      4687
           1       0.70      0.24      0.36      1313
   
    accuracy                           0.81      6000
   macro avg       0.76      0.61      0.62      6000
   weighted avg    0.79      0.81      0.77      6000
   ```

   **Matriz de Confusão:**

   ![Matriz de Confusão](https://github.com/user-attachments/assets/79f614ba-bd20-4609-82c3-b3eef8cf59a9)
  

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

   **Insight:**
      1. **Dados**
         - **Verdadeiro Negativo (TN = 4551):** O modelo classificou corretamente 4551 clientes como não inadimplentes.
         - **Falso Positivo (FP = 136):** 136 clientes foram incorretamente classificados como inadimplentes quando não eram.
         - **Falso Negativo (FN = 997):** 997 clientes que realmente eram inadimplentes foram classificados incorretamente como não inadimplentes.
         - **Verdadeiro Positivo (TP = 316):** O modelo classificou corretamente 316 clientes como inadimplentes.
      
      2. **Métricas de Desempenho:**
         - **Acurácia:** O modelo tem uma acurácia de 81%, o que indica que 81% das previsões feitas pelo modelo estavam corretas. 
         - **Precisão (Classe 1 = Inadimplente):** A precisão para a classe inadimplente é de 70%, o que significa que, das previsões feitas para a classe 1 (inadimplente), 70% estavam corretas.
         - **Recall (Classe 1 = Inadimplente):** O recall é relativamente baixo para a classe 1, com um valor de 24%, indicando que o modelo conseguiu identificar corretamente apenas 24% dos inadimplentes reais.
         - **F1-Score (Classe 1 = Inadimplente):** O F1-score para a classe 1 é de 0.36, sugerindo um equilíbrio moderado entre precisão e recall, mas é mais inclinado a penalizar o baixo recall.
      
      3. **Análise dos Resultados:**
         - **Desempenho no Identificação de Inadimplentes:** O modelo é bom em identificar clientes que não são inadimplentes (alta precisão e recall para a classe 0), mas tem um desempenho limitado em identificar corretamente os inadimplentes (classe 1), conforme evidenciado pelo recall baixo de 24%.
         - **Implicações do Falso Negativo:** O alto número de falsos negativos (997) é preocupante, pois implica que o modelo frequentemente deixa de identificar clientes que realmente são inadimplentes, o que pode levar a um subestimar do risco de crédito.
         - **Trade-off entre Precisão e Recall:** O modelo parece ter uma ligeira tendência a classificar os clientes como não inadimplentes, o que poderia ser um reflexo da distribuição desbalanceada das classes, onde a maioria dos clientes é não inadimplente.


   
3. **Random Forest:**

   **Acurácia:** 0.81

   **Relatório de Classificação:**

   ```plaintext
              precision    recall  f1-score   support

           0       0.84      0.94      0.89      4687
           1       0.63      0.36      0.46      1313

    accuracy                           0.81      6000
   macro avg       0.73      0.65      0.67      6000
   weighted avg    0.79      0.81      0.79      6000
   ```

   **Matriz de Confusão:**

    ![Random Forest](https://github.com/user-attachments/assets/bd4bd901-a804-408e-9c44-12f7ea49975a)

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

   **Insight:** 
    - O modelo tem uma acurácia de 82%, indicando que a maioria das previsões está correta.
   
      - Desempenho Desbalanceado: O recall para a classe de inadimplentes (1) é relativamente baixo (0.64), sugerindo que o modelo tem dificuldade em identificar todos os clientes inadimplentes.
   
      - Falsos Negativos: Há um número significativo de falsos negativos (346), o que pode ser preocupante, pois esses clientes inadimplentes não foram identificados corretamente.
   
      - Falsos Positivos: O número de falsos positivos (280) é menor, mas ainda relevante, indicando que alguns clientes não inadimplentes foram incorretamente classificados como inadimplentes.


Falsos Positivos: O número de falsos positivos (280) é menor, mas ainda relevante, indicando que alguns clientes não inadimplentes foram incorretamente classificados como inadimplentes.

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


```

---
