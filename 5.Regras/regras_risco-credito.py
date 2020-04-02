import Orange

base = Orange.data.Table('../data/risco_credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base)

for regras in classificador.rule_list:
    print(regras)

#história boa,divida alta,garantias nenhuma,renda > 35
#história ruim,divida alta,garantias adequada,renda < 15
resultado1 = classificador(['boa','alta','nenhuma','acima_35'])
resultado2 = classificador(['ruim','alta','adequada','0_15'])
print(base.domain.class_var.values[resultado1])
print(base.domain.class_var.values[resultado2])