import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

#ex1

#a
housing = pd.read_csv('BostonHousing.csv')
medv = housing['medv'].values
rm = housing['rm'].values
crim = housing['crim'].values
indus = housing['indus'].values
#b
with pm.Model() as housing_model:
  α = pm.Normal('α', mu=0, sigma=1000)
  beta_rm = pm.Normal('beta_rm', mu=0, sd=10)
  beta_crim = pm.Normal('beta_crim', mu=0, sd=10)
  beta_indus = pm.Normal('beta_indus', mu=0, sd=10)
  #modelul liniar
  mu = beta_rm * rm + beta_crim * crim + beta_indus * indus
  medv_pred = pm.Normal('medv', mu=mu, sd=1, observed='medv')

  idata_mlr = pm.sample(1250, return_inferencedata=True)

#c
az.plot_forest(idata_mlr,hdi_prob=0.95,var_names=[beta_rm, beta_crim, beta_indus])
az.summary(idata_mlr,hdi_prob=0.95,var_names=[beta_rm, beta_crim, beta_indus])

#d
posterior_g = idata_mlr.posterior.stack(samples={"chain", "draw"}) #avem 5000 de extrageri in esantion (nr. draws x nr. chains)
mu = posterior_g['α']+33*posterior_g[beta_rm, beta_crim, beta_indus][0]+np.log(540)*posterior_g[beta_rm, beta_crim, beta_indus][1]
az.plot_posterior(mu.values,hdi_prob=0.5)


#ex2
#a
def posterior_grid(grid_points=50, heads=4, tails=1):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) # uniform prior
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (4,1)) #de 4 ori ban, o data stema
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')

#b
#Conform figurii 1, valoarea lui theta care maximimeaza probabilitatea este aproximativ 0.2