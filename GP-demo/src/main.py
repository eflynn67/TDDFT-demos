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

psi_func = wf.getPsi_x(0,1,0) ## GS harmonic oscillator.

psiArr = psi_func(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H_func = matrix.get_H_func(name='gaussian')
E_GS, evect_GS = solver.MatrixSolve_SC(H_func, psiArr, psiStarArr)
evect_GS = evect_GS/np.linalg.norm(evect_GS)

## remake the Hamiltonian with GS used in the non-linear interaction
H = H_func(evect_GS,np.conjugate(evect_GS),mass,alpha,q)
print('Numerical GP:', E_GS)
plt.plot(grid,np.real(evect_GS))
plt.plot(grid ,np.imag(evect_GS))
plt.title('Ground state')
plt.show()

V = pots.V_gaussian(grid)
plt.plot(grid,V,label='V')
plt.plot(grid,np.real(evect_GS),label='GS real')
plt.plot(grid,np.imag(evect_GS),label='GS imag')

#plt.xlim([-10,10])
#plt.ylim([-1,1])
plt.title('Potential')
plt.legend()
plt.show()


### Propagation 
psi_series_forward = np.zeros((nt_steps+1,len(grid)),dtype=complex)
psi_series_backward = np.zeros((nt_steps+1,len(grid)),dtype=complex)
psi_series_forward[0] = evect_GS.copy()
psi_series_backward[0] = evect_GS.copy()
for j in range(nt_steps):
    # predictor step
    psi_series_forward[j+1] = solver.prop(psi_series_forward[j],H,0.5*1.0j*delta_t)
    psi_series_backward[j+1] = solver.prop(psi_series_backward[j],H,-0.5*1.0j*delta_t)
    ## recalculate the H using the predictor GS wavefunction
    H = H_func(psi_series_forward[j+1],psi_series_backward[j+1],mass,alpha,q)
    # now calculate the corrector step with the newly calculated H
    psi_series_forward[j+1] = solver.prop(psi_series_forward[j],H,1.0j*delta_t)
    psi_series_backward[j+1] = solver.prop(psi_series_backward[j],H,-1.0j*delta_t)
    ## plot each time step
    #plt.plot(grid,np.real(psi_series_backward[j+1]),label='$Re(\psi)$ backward')
    #plt.plot(grid,np.imag(psi_series_backward[j+1]),label='$Im(\psi)$ backward')
    #plt.plot(grid,np.real(psi_series_forward[j+1]),label='$Re(\psi)$ forward')
    #plt.plot(grid,np.imag(psi_series_forward[j+1]),label='$Im(\psi)$ forward')
    plt.plot(grid,np.real(np.conjugate(psi_series_forward[j])*psi_series_forward[j]),label='$\psi^{*}\psi$')
    plt.plot(grid,V,label='V')
    plt.title(f't = {np.around(delta_t + delta_t*j,4)} (step {j})')
    plt.ylim([-.1,.2 ])
    plt.legend()
    plt.show()
    
print('End')