import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('./dummy.csv')
# ex 2
# dummy_data = np.random.rand(500, 2)
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
#order = 2
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
# plt.scatter(x_1s[0], y_1s)
# plt.xlabel('x')
# plt.ylabel('y')

with pm.Model() as model_l:
  α = pm.Normal('α', mu=0, sigma=1)
  β = pm.Normal('β', mu=0, sigma=10)
  ε = pm.HalfNormal('ε', 5)
  μ = α + β * x_1s[0]
  y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
  idata_l = pm.sample(2000, return_inferencedata=True)
with pm.Model() as model_p:
  α = pm.Normal('α', mu=0, sigma=1)
  # β = pm.Normal('β', mu=0, sigma=10, shape=order)
  # ex 1.b)
  β = pm.Normal('β', mu=0, sigma=100, shape=order)
  # β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
  ε = pm.HalfNormal('ε', 5)
  μ = α + pm.math.dot(β, x_1s)
  y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
  idata_p = pm.sample(2000, return_inferencedata=True)

# ex 1.a)
with model_p:
  pm.set_data({'x_1s': x_1s})
  y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
  pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
  plt.show()


#ex 3
order_cubic = 3
x_1p_cubic = np.vstack([x_1**i for i in range(1, order_cubic+1)])
x_1s_cubic = (x_1p_cubic - x_1p_cubic.mean(axis=1, keepdims=True))
x_1p_cubic.std(axis=1, keepdims=True)
y_1s_cubic = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_cubic:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order_cubic)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s_cubic)
    y_pred_cubic = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s_cubic)
    idata_c = pm.sample(2000, return_inferencedata=True)

waic_cubic = az.waic(idata_c)
loo_cubic = az.loo(idata_c)
print("WAIC - Cubic:", waic_cubic)
print("LOO - Cubic:", loo_cubic)

waic_linear = az.waic(idata_l)
loo_linear = az.loo(idata_l)
print("WAIC - Linear:", waic_linear)
print("LOO - Linear:", loo_linear)

waic_quadratic = az.waic(idata_p)
loo_quadratic = az.loo(idata_p)
print("WAIC - Quadratic:", waic_quadratic)
print("LOO - Quadratic:", loo_quadratic)

az.plot_compare({'Linear': idata_l, 'Quadratic': idata_p, 'Cubic': idata_c})