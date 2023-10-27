import scipy.stats
import numpy as np

lambda_poisson = 20
normal_mean = 2
std_dev = 0.5
poisson_dist = scipy.stats.poisson(lambda_poisson)

num_clients = poisson_dist.rvs()    #clienti pe ora

print("Numar de clienti intr-o ora: " + str(num_clients))

def estimate_alpha():
    alpha_values = np.linspace(1,15,1000)
    final_alpha = 1
    for alpha in alpha_values:
        exponential_dist = scipy.stats.expon(scale=alpha)
        normal_dist = scipy.stats.norm(loc=normal_mean, scale=std_dev)
        
        cook_time = exponential_dist.rvs()
        order_time = normal_dist.rvs()
        total_time = order_time + cook_time

        if np.percentile(total_time, 95) <= 15:
            final_alpha = alpha
        
    return final_alpha

max_alpha = estimate_alpha()
print("Alpha maxim: " + str(max_alpha))

