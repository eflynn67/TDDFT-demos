import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

from init import *
from inputs import *
import matrix
import wf
import potentials as pots
import solver



psi = wf.getPsi_x(0,1,-2) ## GS harmonic oscillator.

psiArr = psi(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H = matrix.construct_H_GP_quartic(psiArr,psiStarArr,mass,alpha,q)

E_GS,psi_static_sol = solver.MatrixSolve_SC(H,name='GP2')
print('Numerical GP:', E_GS)
plt.plot(grid,evect,label=f'GP q = {q}')
plt.legend()
plt.show()




