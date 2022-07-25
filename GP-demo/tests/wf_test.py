import numpy as np
import sys 
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from inputs import *
from init import *
import potentials as pots
import wf


psi = wf.getPsi_x(0,1,-1) ## GS harmonic oscillator.

psiArr = psi(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi

plt.plot(grid,psiArr)
plt.show()