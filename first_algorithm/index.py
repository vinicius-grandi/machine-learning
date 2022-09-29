import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

arquivo = pd.read_csv(
    'C:/Users/breno/OneDrive/Documentos/machine-learning/dataset/wine_dataset.csv'
)

arquivo.head()

# muda os valores da coluna que diz se o vinho é branco ou tinto para 0 e 1.
# 0 significa tinto
# 1 significa branco
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

# pega a coluna style - variável alvo
y = arquivo['style']
# pega o resto - preditoras
x = arquivo.drop('style', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print('Acurácia:', resultado)
