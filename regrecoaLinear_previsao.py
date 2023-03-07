# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Criando um conjunto de dados de exemplo
x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x, y)

# Fazendo previsões usando o modelo
x_novo = np.array([6]).reshape((-1, 1))
previsao = modelo.predict(x_novo)

# Exibindo as previsões
print("O proximo numero pode ser: ",previsao)

"""Neste exemplo, estamos criando um modelo de regressão linear simples para prever os valores de y com base nos valores de x. 
Primeiro, importamos as bibliotecas necessárias e criamos um conjunto de dados de exemplo. 
Em seguida, criamos e treinamos o modelo usando a classe LinearRegression da biblioteca Scikit-Learn.
 Por fim, usamos o modelo treinado para fazer uma previsão para um novo valor de x e exibimos o resultado."""