"""
Created on Tue Oct 15 18:26:10 2019

@author: alexa
"""
primes_cache = []
def prime(n):
    if n == 1: 
        return False
    if n in primes_cache:
        return True
    else:
        for i in primes_cache:
            if n%i == 0:
                return False
        primes_cache.append(n)
        return True
    
import numpy as np
import matplotlib.pyplot as plt

n_points = 100000
n = np.arange(1,n_points+1)
r = 1
k = 1
x = n*np.cos(n)
y = n*np.sin(n)

x_p = []
y_p = []
x_perf = []
y_perf = []

for i in range(1,n_points+1):
    if prime(i):
        x_p.append(x[i-1])
        y_p.append(y[i-1])
    if int(np.sqrt(i))**2==i:
        x_perf.append(x[i-1])
        y_perf.append(y[i-1])

plt.figure(figsize=(12,12))
ax = plt.gca()
ax.set_facecolor((0,0,0))
plt.scatter(x_p,y_p,s=1,c='w')
#plt.scatter(x_perf,y_perf,s=10,c='g'
#plt.plot(x,y,'k')
ax.set_aspect('equal')
ax.set_xlim((-50000,50000))
ax.set_ylim((-50000,50000))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('Prime spiral')
plt.show()