'''clientid:id do cliente / income:salário ou renda(anual) /
age:idade / loan:impréstimo /default:atributo classe/'''
import pandas as pd  # pd é o apelido para pandas
base = pd.read_csv('credit-data.csv')  # carregar a base csv
print(base.describe(), "\n")  # ver algumas estatísticas

'''count:quantidade
mean:media
std:desvio padrão'''

# localiza as idades 'erradas' no caso negativas
print('\n', base.loc[base['age'] < 0], '\n')

'''metodos para modificar dados'''
# apagar a coluna   base.drop('age', 1, inplace=True)
# apagar somente os registro com problema  base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.loc[base.age < 0, 'age'] = (base['age'][base.age > 0].mean())
