import pandas as pd #Importando o Pandas
from sklearn.model_selection import train_test_split #Importando o teste
import matplotlib.pyplot as plt #Importando a biblioteca gráfica
from sklearn.metrics import confusion_matrix #Importando matriz de confusão
from sklearn import tree #Importando a árvore
from sklearn.datasets import load_iris #Importando o DataSet Iris
data = load_iris()

#Transformando o DataSet em um DataFrame
iris = pd.DataFrame(data.data)
iris.columns = data.feature_names #renomeando as colunas do DF para ficar igual a do DS
iris['target'] = data.target  #Adicionando uma coluna target

#Selecionando apenas as colunas de pétalas e esses targets
irisl = iris.loc[iris.target.isin([1,2]), ['petal length (cm)', 'petal width (cm)','target']]

#Definindo x e y
x = irisl.drop('target', axis=1)
y = irisl.target

#teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# plot
fig, ax = plt.subplots()
ax.scatter(X_train['petal length (cm)'], X_train['petal width (cm)'], c=y_train)

clf = tree.DecisionTreeClassifier(random_state=42)#Criando um classificador
clf = clf.fit(X_train, y_train)#fit com os dados de treino
clf.score(X_train, y_train)#score
tree.plot_tree(clf) #Plot da árvore

#Fazendo a previsão e avaliação do erro
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)


