import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import assoc_laguerre
sys.path.insert(0, '../src')
import wf
import densities
from init import *
'''
psi_array = np.zeros((2,nmax,lmax+1,len(spin),len(grid)))
for q in range(2):
    for n in range(nmax):
        for l in range(lmax+1):
            for s in range(len(spin)):
                print(q,n,l,s)
                psi_func = wf.get_wfHO_radial(n, l)#wf.initWfs(name='hydrogen',n=0,l=0)
                eval_psi = psi_func(grid)
                psi_array[q,n,l,s] = eval_psi/np.linalg.norm(eval_psi)
'''
psi_array = wf.initWfs(N,Z,name='HO')
plt.plot(grid,psi_array[0][0][1][1])
plt.title('a wavefunction')
plt.show()


rho = densities.rho(psi_array,grid)

plt.plot(grid,rho[0])
plt.title('total rho')
plt.show()

plt.title('proton rho')
rho = densities.rho(psi_array,grid)
plt.plot(grid,rho[1])
plt.show()

plt.title('neutron rho')
rho = densities.rho(psi_array,grid)
plt.plot(grid,rho[1])
plt.show()
