from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'I'), ('I', 'A'), ('C', 'A')])

#cutremur
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])

#incendiu
cpd_i = TabularCPD(variable='I', variable_card=2, 
                            #C=0  C=1
                   values=[[0.99, 0.97],#not I
                           [0.01, 0.03]],#I
                   evidence=['C'], 
                   evidence_card=[2])

#alarma
cpd_a = TabularCPD(variable='A', variable_card=2, 
                         #C=I=0;C=1,I=0;C=0,I=1;C=I=1
                   values=[[0.9999, 0.98, 0.05, 0.02], #not A
                           [0.0001, 0.02, 0.95, 0.98]],#A
                   evidence=['C', 'I'],
                   evidence_card=[2, 2])

model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()

infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)

result = infer.query(variables=['I'], evidence={'A': 0})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()