''' age:idade / workclass:tipo de emprego/final_weight:caracteristica demográfica e socioeconomica
education:nivel de educaçaõ/education-num:anos que a pessoa estudou ou está estudando /marital-status:
estado civil/ocupation:ocupação/relationship:relacionamento/race:raça/sex:sexo/capital-gain:ganho de capital
/capital-loops:perda de capital/hour-per-week:horas de trabalho semanais/native-country:pais nativo/income:renda(classe)'''

import pandas as pd
import numpy as np
base = pd.read_csv('census.csv')
'''Transformação de variáveis categóricas'''

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
''' Passando as variáveis categóricas nominais e ordinais para Numéricas'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:, 1])

previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
'''Transformações das variaveis categoricas para estilo 'dummy' '''

onehotencoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13],
                              handle_unknown='ignore')

onehotencoder = OneHotEncoder(handle_unknown='ignore')

previsores = onehotencoder.fit_transform(previsores).numpy.asarray()