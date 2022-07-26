import numpy as np
import sys 
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from inputs import *
from init import *
import potentials as pots
import wf

psi = wf.getPsi_x(0,1,0) ## GS harmonic oscillator.

psiArr = psi(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi

V = pots.V_gaussian(grid) + pots.V_rho(psiArr, np.conjugate(psiArr), q)

plt.plot(grid,V)
plt.plot(grid,psiArr)
plt.xlim([-5,5])
plt.ylim([-2,2])
plt.show()
