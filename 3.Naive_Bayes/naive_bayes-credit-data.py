import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

''' Divisão de bases de dados'''
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# importação da biblioteca
# criação do classificador
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
#precisao = accuracy_score(classe_teste, previsoes)
#matriz = confusion_matrix(classe_teste, previsoes)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento,classe_treinamento)
previsoes = classificador.predict(previsores_teste) 

from sklearn.metrics import  confusion_matrix,accuracy_score
previsao = accuracy_score(classe_teste,previsoes)#percentual de acerto

matriz = confusion_matrix(classe_teste,previsoes)
print(matriz)