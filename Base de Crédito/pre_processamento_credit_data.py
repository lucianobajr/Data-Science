'''clientid:id do cliente / income:salário ou renda(anual) /
age:idade / loan:impréstimo /default:atributo classe/'''
import pandas as pd
base = pd.read_csv('credit-data.csv')
print(base)
print(base.describe())
'''
count:quantidade
mean:media
std:desvio padrão
'''
