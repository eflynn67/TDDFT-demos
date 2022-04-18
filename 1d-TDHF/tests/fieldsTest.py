import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import wf
import densities
import fields 
from init import *

psi_array = wf.initWfs(N,Z,name='HO')
plt.plot(grid,psi_array[0][0][1][1])
plt.title('a wavefunction')
plt.show()


rho = densities.rho(psi_array,grid[10])
yuk_eval = fields.yuk(psi_array,grid[10])
#plt.plot(grid,yuk_eval)
#plt.show()