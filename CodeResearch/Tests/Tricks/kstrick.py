import math

from scipy.stats import ksone, kstwobign

n = 100
D = 0.15

def calcKolmogorov(t, k=1000):

    if t < 0:
        return 0

    sum = 0.0

    for i in range(-k, k + 1, 1):
        d = ((-1)**i) * math.exp(-2*i*i*t*t)
        print(d)
        sum += d

    return sum

def calcSimpleKolmogorov(t, n):
    return t + 1/(6*math.sqrt(n)) + (t - 1) / (4 * n)

p1 = ksone.cdf(D, n)
p2 = calcKolmogorov(D * math.sqrt(n), 16)
p3 = kstwobign.cdf(D * math.sqrt(n))

print(0.3 * math.sqrt(1000))

print(f'p1 = {p1}, p2 = {p2}, p3 = {p3}')