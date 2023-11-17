import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np
############ a)

#citim datele si le curatam
data = pd.read_csv('auto-mpg.csv').dropna(subset=['horsepower', 'mpg'])
data = data[data['horsepower'] != '?']
data = data[data['mpg'] != '?']
#relatia dintre CP si mpg
mpg = data['mpg'].values
cp = data['horsepower'].values
plt.figure(figsize=(15, 15))
plt.xticks(fontsize=6)
plt.xticks(rotation=45)
plt.scatter(cp,mpg)
plt.xlabel('Cai putere')
plt.ylabel('Mile/galon')
plt.title('Relatia dintre CP si mpg')
plt.show()

########## b)

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)

    #modelul de regresie liniara simpla
    mu = alpha + beta * cp

    sigma = pm.HalfCauchy('sigma', sigma=10)

    mpg = pm.Normal('mpg', mu=mu, sigma=sigma, observed=mpg)
    
    trace = pm.sample(100, tune=100)

az.plot_trace(trace, var_names=['alpha', 'beta', 'sigma'])
plt.show()

#din modelul anterion gasim estimarea MAP
with model:
    map_estimate = pm.find_MAP()

alpha_map = map_estimate['alpha']
beta_map = map_estimate['beta']

plt.scatter(data['horsepower'], data['mpg'])
plt.plot(data['horsepower'], alpha_map + beta_map * data['horsepower'], color='red', label='Dreapta de regresie')
plt.xlabel('Cai putere')
plt.ylabel('Mile/galon')
plt.legend()
plt.show()