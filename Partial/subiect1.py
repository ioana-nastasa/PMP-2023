from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

#posibilitati:
#primul jucator: stema; al doilea jucator: s/s, s/b, b/s, b/b
#primul jucator: ban; al doilea jucator: stema sau ban

#primul jucator castiga in cazurile: 1,2,3,5
#al doilea jucator castiga in cazurile: 0,4
#0: 1/2 * 1/3 * 1/3 = 0.05
#1: 1/2 * 1/3 * 2/3 = 0.11
#2: 1/2 * 2/3 * 1/3 = 0.11
#3: 1/2 * 2/3 * 2/3 = 0.22
#4: 1/2 * 1/3 = 0.16
#5: 1/2 * 2/3 = 0.33

#prob primul = (0.11+0.11+0.22+0.33)*(4/6) = 0.66
#prob al doilea = 0.33
model = BayesianNetwork([('s', 'ss'),('s', 'sb'),('s', 'bs'),('s','bb'), ('b', 's'),('b', 'b')])


castiga_primul = TabularCPD(variable='castiga_primul', variable_card=2, values = [[0.66], [0.33]])
castiga_al_doilea = TabularCPD(variable='castiga_al_doilea', variable_card=2, values=[[0.33],[0.66]])

model.add_cpds(castiga_primul, castiga_al_doilea)
assert model.check_model()