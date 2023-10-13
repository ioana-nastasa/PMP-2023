import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

lambda1 = 4
lambda2 = 6

P1 = 0.4
P2 = 0.6

num = 10000

#generam alegerile clientilor pentru mecanici
choice = np.random.choice([1, 2], size = num, p=[P1, P2])

time = np.zeros(num) #vectorul cu timpii generati, initializat cu 0

#generam timpi de asteptare
for i in range(1, num):
    if choice[i] == 1:
        time[i] = stats.expon(scale=1/lambda1).rvs()
    else:
        time[i] = stats.expon(scale=1/lambda2).rvs()

#calculam media E si deviatia standard a timpilor
mean = np.mean(time)
standard_dev = np.std(time)

print("Media: " + str(mean))
print("Deviatia standard: " + str(standard_dev))


plt.hist(time, bins=50, density=True, alpha=0.6, color='r')
plt.xlabel('X - Timp de servire')
plt.ylabel('Densitatea')
plt.grid(True)

plt.show()