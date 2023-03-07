# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregando um conjunto de dados de exemplo
dados = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = dados.iloc[:, :-1].values
y = dados.iloc[:, -1].values

# Dividindo o conjunto de dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalizando os dados
sc = StandardScaler()
X_treino = sc.fit_transform(X_treino)
X_teste = sc.transform(X_teste)

# Criando e treinando o modelo de classificação
modelo = LogisticRegression(random_state=0)
modelo.fit(X_treino, y_treino)

# Fazendo previsões usando o modelo
previsoes = modelo.predict(X_teste)

# Avaliando o desempenho do modelo
acuracia = accuracy_score(y_teste, previsoes)
matriz_confusao = confusion_matrix(y_teste, previsoes)

# Exibindo as métricas de avaliação
print('Acurácia:', acuracia)
print('Matriz de confusão:')
print(matriz_confusao)

"""Neste exemplo, estamos criando um modelo de classificação para prever a espécie de uma flor com base em suas características.
   Primeiro, importamos as bibliotecas necessárias e carregamos um conjunto de dados de exemplo da UCI Machine Learning Repository.
   Em seguida, dividimos o conjunto de dados em conjuntos de treino e teste, normalizamos os dados e criamos e treinamos o modelo
   de classificação usando a classe LogisticRegression da biblioteca Scikit-Learn.
   Por fim, usamos o modelo treinado para fazer previsões para os dados de teste e avaliamos o desempenho do modelo usando a 
   acurácia e a matriz de confusão."""
