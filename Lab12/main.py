import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#ex1
def posterior_grid(grid_points=50, observed_heads=6, observed_tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(observed_heads, observed_heads + observed_tails, grid)
    posterior = likelihood*prior
    posterior /= posterior.sum()
    return grid, posterior

observed_data = np.repeat([0, 1],(10, 3))
data_points = 10
observed_heads = observed_data.sum()
observed_tails = len(observed_data) - observed_heads

grid_custom, posterior_custom = posterior_grid(data_points, observed_heads, observed_tails)
plt.plot(grid_custom, posterior_custom, 'o-')
plt.title(f'Heads = {observed_heads}, Tails = {observed_tails}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

#ex2
num_samples = 10000
random_numbers_x, random_numbers_y = np.random.uniform(-1, 1, size=(2, num_samples))
inside_circle=(random_numbers_x**2 + random_numbers_y**2) <= 1
pi = inside_circle.sum() * 4 / num_samples
error_percentage = abs((pi - np.pi) /pi) * 100

outside_circle = np.invert(inside_circle)

plt.figure(figsize=(8, 8))
plt.plot(random_numbers_x[inside_circle], random_numbers_y[inside_circle], 'b.')
plt.plot(random_numbers_x[outside_circle], random_numbers_y[outside_circle], 'r.')
plt.plot(0, 0, label=f'π Estimate = {pi:4.3f}, Error = {error_percentage:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()
