import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import densities as den
import wf
from init import *

psiArr,energiesArr = wf.initWfs(name='HO')

rho = den.rho(psiArr, np.conjugate(psiArr), grid)
plt.plot(grid,rho)
plt.show()