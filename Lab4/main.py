import scipy.stats
import numpy as np

lambda_poisson = 20
normal_mean = 2
std_dev = 0.5

poisson_dist = scipy.stats.poisson(lambda_poisson)
normal_dist = scipy.stats.norm(loc=normal_mean, scale=std_dev)

order_time = normal_dist.rvs()      #timp de plasare comanda si plata
num_clients = poisson_dist.rvs()    #clienti pe ora

print("Numar de clienti intr-o ora: " + str(num_clients))
print("Timp de plasare comanda si plata: " + format(order_time, '.4f') + " min")

def estimate_alpha():
    alpha_values = np.linspace(1,15,1000)
    final_alpha = None

    for alpha in alpha_values:
        exponential_dist = scipy.stats.expon(scale=alpha)   #incercam valori pentru alpha de la 1 la 15

        total_times = []
        for _ in range(num_clients):
            cook_time = exponential_dist.rvs()
            total_time = order_time + cook_time
            total_times.append(total_time)

        if np.percentile(total_times,95) <= 15:
            final_alpha = alpha
        
    return final_alpha

max_alpha = estimate_alpha()
print("Alpha maxim: " + format(max_alpha, '.4f'))

#calculam distributia exponentiala cu alpha aflat
final_exp_dist = scipy.stats.expon(scale=max_alpha)

total_times = []

#avand timpul de plasare comanda si numarul de clienti, generate aleator, simulam timpi de gatire cu alpha aflat

for _ in range(num_clients):
    cook_time = final_exp_dist.rvs()
    total_time = order_time + cook_time
    total_times.append(total_time)

avg_waiting_time = np.mean(total_times)
print("Timp mediu de asteptare: " + format(avg_waiting_time, '.4f') + " min")