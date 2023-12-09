# Importações
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Carregar o dataset
url = "https://raw.githubusercontent.com/BrunoBasstos/mvp-essi/main/.src/titanic-dataset.csv"
data = pd.read_csv(url)

# Selecionar colunas relevantes
colunas = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data = data[colunas]

# Divisão dos dados
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definição de variáveis numéricas e categóricas
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Pclass', 'Sex', 'Embarked']

# Configurações dos modelos
modelos = {
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

parametros = {
    'KNN': {'classifier__n_neighbors': [3, 5, 7, 9]},
    'NB': {},  # Naive Bayes não tem hiperparâmetros relevantes para ajustar neste caso
    'CART': {'classifier__max_depth': [3, 5, 7, None]},
    'SVM': {'classifier__C': [0.1, 1, 10, 100], 'classifier__gamma': ['scale', 'auto']}
}

# Preparação de Pré-processadores
preprocessors = {
    'original': ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)]),
    'padronizado': ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)]),
    'normalizado': ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
}

# Avaliação dos Modelos com Diferentes Pré-processamentos
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
melhores_resultados = {}

for preproc_nome, preproc in preprocessors.items():
    for nome_modelo, modelo in modelos.items():
        pipeline = Pipeline([('preprocessor', preproc), ('classifier', modelo)])
        if parametros[nome_modelo]:
            grid_search = GridSearchCV(pipeline, parametros[nome_modelo], cv=kfold, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            melhor_modelo = grid_search.best_estimator_
            score = grid_search.best_score_
        else:
            melhor_modelo = pipeline
            melhor_modelo.fit(X_train, y_train)
            score = cross_val_score(melhor_modelo, X_train, y_train, cv=kfold, scoring='accuracy').mean()

        print(f"{preproc_nome} - {nome_modelo}: Melhor Score = {score}")
        melhores_resultados[f"{preproc_nome}_{nome_modelo}"] = (melhor_modelo, score)

# Escolher e Exportar o Melhor Modelo Geral
melhor_modelo_geral = max(melhores_resultados.items(), key=lambda x: x[1][1])

# Treinar o melhor modelo geral com todo o conjunto de treinamento
melhor_modelo_geral[1][0].fit(X_train, y_train)

# Avaliar a acurácia do melhor modelo no conjunto de teste
y_pred = melhor_modelo_geral[1][0].predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do melhor modelo no conjunto de teste: {accuracy}")

# Exportar o melhor modelo
joblib.dump(melhor_modelo_geral[1][0], 'melhor_modelo_titanic.pkl')
print(f"Melhor modelo geral ({melhor_modelo_geral[0]}) exportado como 'melhor_modelo_titanic.pkl'")

# Como usar o modelo

# Carregar o melhor modelo
modelo = joblib.load('melhor_modelo_titanic.pkl')

# Teste 1

# Dados de entrada
#Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
entrada1 = {
    'Pclass': 3,
    'Name': 'Mr. Owen Harris Braund',
    'Sex': 'male',
    'Age': 22,
    'SibSp': 1,
    'Parch': 0,
    'Ticket': 'A/5 21171',
    'Fare': 7.25,
    'Cabin': '',
    'Embarked': 'S'
}

# Transformar os dados de entrada em um DataFrame
entrada1 = pd.DataFrame(entrada1, index=[0])

# Fazer a predição
predicao1 = modelo.predict(entrada1)
print(f"Predição 1: {predicao1}")

# Teste 2

# Dados de entrada

entrada2 = {
    'Pclass': 1,
    'Name': 'Mrs. John Bradley (Florence Briggs Thayer) Cumings',
    'Sex': 'female',
    'Age': 38,
    'SibSp': 1,
    'Parch': 0,
    'Ticket': 'PC 17599',
    'Fare': 71.2833,
    'Cabin': 'C85',
    'Embarked': 'C'
}

# Transformar os dados de entrada em um DataFrame
entrada2 = pd.DataFrame(entrada2, index=[0])

# Fazer a predição
predicao2 = modelo.predict(entrada2)
#print(f"Predição 2: {predicao2}")
# informar "sobreviveu" ou "não sobreviveu" em vez de 1 ou 0
print (f"Predição 2: {predicao2[0]}", "sobreviveu" if predicao2[0] == 1 else "não sobreviveu")
