import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import ExtraTreesClassifier

dados = pd.read_csv(r"C:\\Users\\User\\Documents\\BDmachine\\wine_dataset.csv")

print("**********Foram usados 6497 linhas de dados totais com 13 atributos (colunas)")
print("**********Exemplo de dados usados no aprendizado:")
print()
print(dados.head())

dados["style"] = dados["style"].replace("red", 0)
dados["style"] = dados["style"].replace("white", 1)

#separando os dados
y = dados["style"]
x = dados.drop("style", axis = 1)

#conjunto de dadoa para treino e para testes =)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.3)

#criando modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

#imprimir resultados
resultado = modelo.score(x_teste, y_teste)
print()
print("precisao nos testes: ", resultado)
#"precisao:  0.9964102564102564"