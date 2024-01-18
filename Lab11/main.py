import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

#ex1
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs =[2, 2, 2]
mix=np.random.normal(np.repeat(means, n_cluster),np.repeat(std_devs, n_cluster))

az.plot_kde(np.array(mix))
plt.show()

#ex2
for n_components in range(2, 5):
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(n_components))
        means = pm.Normal('means', mu=np.arange(n_components)*5, sd=2, shape=n_components)
        std_devs = pm.HalfNormal('std_devs', sd=2, shape=n_components)
        components = pm.Normal.dist(mu=means, sd=std_devs)
        mix_likelihood = pm.Mixture('mix_likelihood', w=weights, comp_dists=components, observed=mix)
        trace = pm.sample(2000, tune=1000, chains=2)

    pm.plot_posterior(trace)
    plt.show()

    #3
    waic = az.waic(trace)
    loo = az.loo(trace)

    print(f"\nModel cu {n_components} componente:")
    print(f"WAIC: {waic.waic}")
    print(f"LOO: {loo.loo}")
