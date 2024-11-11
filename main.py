import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        print("Estrutura inicial dos dados:")
        print(self.df.head())  # Exibe as primeiras linhas
        print("\nInformações gerais sobre os dados:")
        print(self.df.info())  # Exibe o resumo de cada coluna
        print("\nVerificando valores ausentes:")
        print(self.df.isnull().sum())  # Exibe a contagem de valores ausentes
        
        # Verificar e tratar valores faltantes        
        if self.df.isnull().sum().any():
            self.df.dropna(inplace=True)

        # Converter rótulos de espécies em valores numéricos
        self.df['Species'] = self.df['Species'].astype('category').cat.codes


    def Treinamento(self):
        X = self.df.iloc[:, :-1]
        y = self.df['Species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar SVM
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        self.svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    
        # Treinar Regressão Logística
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_train, y_train)
        self.lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

        dec_tree = DecisionTreeClassifier()
        dec_tree.fit(X_train, y_train)
        self.dectree_accuracy = accuracy_score(y_test, dec_tree.predict(X_test))

    def Teste(self):
        print(f"Acurácia do SVM: {self.svm_accuracy:.2f}")
        print(f"Acurácia da Regressão Logística: {self.lr_accuracy:.2f}")
        print(f"Acurácia da Árvore de Decisões: {self.dectree_accuracy:.2f}")

    def Graficos(self):
        
        # Explorando os gráficos
        
        # Histogramas
        self.df.hist(bins=20, figsize=(10, 8), edgecolor="k")
        plt.suptitle("Distribuição das Características", y=1.02)
        plt.show()

        # Boxplots
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df.drop(columns=['Species']))
        plt.title("Boxplot das Características para Identificação de Outliers")
        plt.show()

        # Pairplot para visualização entre características e espécies
        sns.pairplot(self.df, hue="Species", palette="Set2", diag_kind="kde")
        plt.suptitle("Pairplot das Características por Espécie", y=1.02)
        plt.show()

        # Matriz de Correlação
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', square=True)
        plt.title("Matriz de Correlação")
        plt.show()

    def Train(self):
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.
        self.TratamentoDeDados()
        self.Treinamento()
        self.Graficos()
        self.Teste()  # Exibe as acurácias dos modelos



# Para executar
modelo = Modelo()
modelo.Train()
