import numpy as np
import matplotlib.pyplot as plt

evals = np.loadtxt('evals.txt',delimiter=',')

plt.scatter(-1*evals[0],-1*evals[1])
plt.xlabel(r'Re $\lambda$')
plt.ylabel(r'Im $\lambda$')
plt.xlim([0,10])
plt.show()

n_array = np.arange(0,len(evals[1]),1)
plt.plot(n_array,sorted(evals[1]),'o')
plt.show()
