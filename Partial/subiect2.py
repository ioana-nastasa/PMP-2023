import matplotlib.pyplot as plt
import pymc as pm
from scipy import stats
import arviz as az

mu = 3.0
sigma = 1.0

#am aproximat un timp mediu de 3 minute, cu deviatie standard de 1 minut
times = stats.norm.rvs(mu,sigma,size=200)

with pm.Model():
    x = pm.Normal('x', mu=mu, sigma=sigma)
    trace = pm.sample(200,cores=1)
    az.plot_posterior(trace)
    plt.show()
