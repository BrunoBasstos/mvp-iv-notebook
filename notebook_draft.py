# Configuração para não exibir os warnings
import warnings

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# Importação de bibliotecas necessárias para análise de dados, pré-processamento e modelagem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

### Bloco de Carregamento e Exploração Inicial do Dataset

# Carregar o dataset a partir da URL especificada
url = "https://raw.githubusercontent.com/BrunoBasstos/mvp-essi/main/.src/titanic-dataset.csv"
dataset = pd.read_csv(url, delimiter=',')

# Exibir informações básicas sobre o dataset
dataset.info()

# Exibir estatísticas descritivas do dataset
dataset.describe()

# Exibir as primeiras linhas do dataset para uma visualização inicial
dataset.head()

### Bloco de Preparação dos Dados

# Definir as variáveis numéricas e categóricas
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Pclass', 'Sex', 'Embarked']

# Criar transformadores para tratamento de dados numéricos e categóricos
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Preencher valores faltantes com a mediana
    ('scaler', StandardScaler())  # Normalizar os dados
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Preencher valores faltantes com 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Aplicar codificação one-hot
])

# Combinar os transformadores em um pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),  # Aplicar transformações em variáveis numéricas
        ('cat', cat_transformer, cat_features)  # Aplicar transformações em variáveis categóricas
    ])

# Preparar os dados para modelagem
X = dataset.drop('Survived', axis=1)  # Separar variáveis independentes
y = dataset['Survived']  # Separar a variável dependente (alvo)
X_preprocessed = preprocessor.fit_transform(X)  # Aplicar o pré-processamento

### Bloco de Divisão do Dataset em Treino e Teste

# Definir tamanho do conjunto de teste e seed para reprodutibilidade
test_size = 0.20
seed = 7

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y,
                                                    test_size=test_size, shuffle=True, random_state=seed, stratify=y)

# Definir parâmetros para a validação cruzada
scoring = 'accuracy'
num_particoes = 10
kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, random_state=seed)

### Bloco de Modelagem e Avaliação de Modelos

# Definir uma semente global para reprodutibilidade
np.random.seed(seed)

# Listas para armazenar os armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
optimized_results = []
names = []

# Criando os elementos do pipeline

# Algoritmos que serão utilizados
knn = ('KNN', KNeighborsClassifier())
cart = ('CART', DecisionTreeClassifier())
naive_bayes = ('NB', GaussianNB())
svm = ('SVM', SVC())

# Transformações que serão utilizadas
standard_scaler = ('StandardScaler', StandardScaler())
min_max_scaler = ('MinMaxScaler', MinMaxScaler())

# Montando os pipelines

# Dataset original
pipelines.append(('KNN-orig', Pipeline([knn])))
pipelines.append(('CART-orig', Pipeline([cart])))
pipelines.append(('NB-orig', Pipeline([naive_bayes])))
pipelines.append(('SVM-orig', Pipeline([svm])))

# Dataset Padronizado
pipelines.append(('KNN-padr', Pipeline([standard_scaler, knn])))
pipelines.append(('CART-padr', Pipeline([standard_scaler, cart])))
pipelines.append(('NB-padr', Pipeline([standard_scaler, naive_bayes])))
pipelines.append(('SVM-padr', Pipeline([standard_scaler, svm])))

# Dataset Normalizado
pipelines.append(('KNN-norm', Pipeline([min_max_scaler, knn])))
pipelines.append(('CART-norm', Pipeline([min_max_scaler, cart])))
pipelines.append(('NB-norm', Pipeline([min_max_scaler, naive_bayes])))
pipelines.append(('SVM-norm', Pipeline([min_max_scaler, svm])))

# Armazenar resultados de desempenho
resultados_desempenho = {}

# Avaliação Inicial dos Modelos com Pipelines
for name, model in pipelines:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    resultados_desempenho[name] = cv_results.mean()
    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # print(msg)

# Boxplot de comparação dos modelos
fig = plt.figure(figsize=(25, 6))
fig.suptitle('Comparação dos Modelos - Dataset Original, Padronizado e Normalizado')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=90)
plt.show()

modelos_ordenados = sorted(resultados_desempenho.items(), key=lambda x: x[1], reverse=True)

print(f"Resultados de Desempenho dos Modelos: {resultados_desempenho}\n")

modelos_selecionados = [model[0] for model in modelos_ordenados[:2]]

# Definição de Parâmetros de Otimização para os Modelos
parametros_otimizacao = {
    'KNN': {
        'modelo': KNeighborsClassifier(),
        'parametros': {'n_neighbors': [3, 5, 7, 9], 'metric': ["euclidean", "manhattan", "minkowski"],
                       'weights': ['uniform', 'distance']}
    },
    'CART': {
        'modelo': DecisionTreeClassifier(),
        'parametros': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [3, 5, 7, 9]}
    },
    'NB': {
        'modelo': GaussianNB(),
        'parametros': {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06], 'priors': [None, [0.5, 0.5], [0.3, 0.7]]}
    },
    'SVM': {
        'modelo': SVC(),
        'parametros': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'gamma': ['scale', 'auto']}
    }
}

# Função para Otimização de Modelo
def otimiza_modelo(nome_modelo, X, y, parametros, cv):
    nome_modelo_lista = nome_modelo.split('-')[0]
    modelo = parametros[nome_modelo_lista]['modelo']
    param_grid = parametros[nome_modelo_lista]['parametros']
    grid = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_


# Lista de Métricas para Avaliação
metricas = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}

# Avaliação e Otimização dos Modelos Selecionados

for nome_modelo in modelos_selecionados:
    melhor_modelo, melhor_score = otimiza_modelo(nome_modelo, X_train, y_train, parametros_otimizacao, kfold)
    optimized_results.append(melhor_score)
    print(f"Melhor Modelo para {nome_modelo}: {melhor_modelo}, Score: {melhor_score}")

    # Avaliação com Múltiplas Métricas
    resultados_metricas = {}
    for metrica_nome, metrica_func in metricas.items():
        score = cross_val_score(melhor_modelo, X_train, y_train, cv=kfold, scoring=make_scorer(metrica_func)).mean()
        resultados_metricas[metrica_nome] = score
    print(f"Resultados das Métricas para {nome_modelo}: {resultados_metricas}\n")


# Boxplot de comparação dos modelos otimizados
fig = plt.figure(figsize=(25, 6))
fig.suptitle('Comparação dos Modelos Otimizados')
ax = fig.add_subplot(111)
plt.boxplot(optimized_results)
ax.set_xticklabels(optimized_results[:4], rotation=90)
plt.show()

