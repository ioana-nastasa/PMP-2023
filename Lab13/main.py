import arviz as az
import matplotlib.pyplot as plt

data_centered = az.load_arviz_data("centered_eight")
data_non_centered = az.load_arviz_data("non_centered_eight")

#ex1
#modelul centrat
num_chains_centered = data_centered.posterior.chain.size
total_samples_centered = data_centered.posterior.draw.size
print("Modelul centrat:")
print(f"Numar de lanturi: {num_chains_centered}, Marimea totala a esantionului: {total_samples_centered}")
az.plot_posterior(data_centered, var_names=['mu', 'tau'], figsize=(10, 6), textsize=14, round_to=2, hdi_prob=0.95, rope=[-0.1, 0.1])

#modelul necentrat
print("\nModelul necentrat:")
num_chains_non_centered = data_non_centered.posterior.chain.size
total_samples_non_centered = data_non_centered.posterior.draw.size
print(f"Numar de lanturi: {num_chains_non_centered}, Marimea totala a esantionului: {total_samples_non_centered}")
az.plot_posterior(data_non_centered, var_names=['mu', 'tau'], figsize=(10, 6), textsize=14, round_to=2, hdi_prob=0.95, rope=[-0.1, 0.1])

plt.show()


#ex2
rhat_centered = az.rhat(data_centered, var_names=['mu', 'tau'])
rhat_non_centered = az.rhat(data_non_centered, var_names=['mu', 'tau'])
print("R-hat pentru modelul centrat:\n", rhat_centered)
print("R-hat pentru modelul necentrat:\n", rhat_non_centered)

az.plot_autocorr(data_centered, var_names=['mu', 'tau'])
az.plot_autocorr(data_non_centered, var_names=['mu', 'tau'])
plt.show()

#ex3
#centrat
divergences_centered = data_centered.sample_stats["diverging"].sum()
print(f"Numarul de divergente pentru modelul centrat: {divergences_centered}")
az.plot_pair(data_centered, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model centrat: divergente")
plt.show()

#necentrat
divergences_non_centered = data_non_centered.sample_stats["diverging"].sum()
print(f"Numarul de divergente pentru modelul necentrat: {divergences_non_centered}")
az.plot_pair(data_non_centered, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model necentrat: divergente")
plt.show()
