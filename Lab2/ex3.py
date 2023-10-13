import random
import matplotlib.pyplot as plt

#numarul de experimente
results = 100

#numarul de aruncari
tosses = 10

#probabilitatea celei de-a doua monezi
p_tails = 0.3

#initializam cu 0 listele ce memoreaza fiecare rezultat
tt_cnt = [0 for _ in range(tosses + 1)] #stema stema
th_cnt = [0 for _ in range(tosses + 1)] #stema ban
ht_cnt = [0 for _ in range(tosses + 1)] #ban stema
hh_cnt = [0 for _ in range(tosses + 1)] #ban ban

for _ in range(results):
    for _ in range(tosses):
        #0 pentru ban, 1 pentru stema
        # prima aruncare
        coin1 = random.randint(0, 1)
        # a doua aruncare, cu probabilitatea stemei p_tails (0.3)
        coin2 = 1 if random.random() < p_tails else 0
        if coin1 == 0 and coin2 == 0:
            tt_cnt[_] += 1
        elif coin1 == 0 and coin2 == 1:
            th_cnt[_] += 1
        elif coin1 == 1 and coin2 == 0:
            ht_cnt[_] += 1
        else:
            hh_cnt[_] += 1

x = list(range(tosses + 1))
plt.plot(x, tt_cnt, label='stema stema')
plt.plot(x, th_cnt, label='stema stema')
plt.plot(x, ht_cnt, label='ban stema')
plt.plot(x, hh_cnt, label='ban ban')

plt.xlabel('Numarul de aruncari')
plt.ylabel('Numar aparitii')
plt.legend()
plt.title('Distributia variabilelor aleatoare')
plt.show()
