'''clientid:id do cliente / income:salário ou renda(anual) /
age:idade / loan:impréstimo /default:atributo classe/'''
import pandas as pd  # pd é o apelido para pandas
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit-data.csv')  # carregar a base csv
print(base.describe(), "\n")  # ver algumas estatísticas
'''count:quantidade
mean:media
std:desvio padrão'''

# localiza as idades 'erradas' no caso negativas
print('\n', base.loc[base['age'] < 0], '\n')
'''metodos para modificar dados-Tratamento de valores inconsistentes'''
# apagar a coluna
# base.drop('age', 1, inplace=True)
# apagar somente os registro com problema
# base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
#base.loc[base.age < 0, 'age'] = (base['age'][base.age > 0].mean())
''' Tratamento  de valores faltantes'''
#imprime valores não preenchidos
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#devemos fazer uma separação dos atributos previsores e outra do atributo classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
''' Escalonamento de Atributos '''
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)