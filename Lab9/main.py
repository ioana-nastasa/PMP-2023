import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Admission.csv')

admitted = data['Admission'].values
gre = data['GRE'].values
gpa = data['GPA'].values

#ex 1
def logistic(x):
    return 1 / (1 + np.exp(-x))

with pm.Model() as logistic_model:
    beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=10)

    miu = beta_0 + beta_1 * gre + beta_2 * gpa

    admit_prob = pm.Deterministic('admit_prob', logistic(miu))
    admission_obs = pm.Bernoulli('admission_obs', p=admit_prob, observed=admitted)

with logistic_model:
    trace = pm.sample(2000, tune=1000)

#ex 2
decision_boundary = -trace['beta_0'] / trace['beta_1']

hdi = pm.hpd(decision_boundary, hdi_prob=0.94)

print("Granita de decizie: " + str(np.mean(decision_boundary)))
print("Intervalul 94% HDI: " + str(hdi))

plt.hist(decision_boundary, bins=30, alpha=0.5, color='skyblue')
plt.axvline(np.mean(decision_boundary), color='orange', linestyle='-', lw=2)
plt.axvline(hdi[0], color='black', linestyle='--', lw=2)
plt.axvline(hdi[1], color='black', linestyle='--', lw=2)
plt.xlabel('Granita de decizie')
plt.ylabel('Frecventa')
plt.show()

#ex 3
stud1_gre = 550
stud1_gpa = 3.5

p = trace['beta_0'] + trace['beta_1'] * stud1_gre + trace['beta_2'] * stud1_gpa

probs = logistic(p)

hdi_prob_stud1 = pm.hpd(probs, hdi_prob=0.9)

print("90% HDI pentru probabilitatea de a fi admis studentul 1:" + str(hdi_prob_stud1))

#ex 4
stud2_gre = 500
stud2_gpa = 3.2

p_stud2 = trace['beta_0'] + trace['beta_1'] * stud2_gre + trace['beta_2'] * stud2_gpa

probs_stud2 = logistic(p_stud2)

hdi_prob_stud2 = pm.hpd(probs_stud2, hdi_prob=0.9)

print("90% HDI pentru probabilitatea de a fi admis studentul 2:" + str(hdi_prob_stud2))