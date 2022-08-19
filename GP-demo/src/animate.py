import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from init import *
from inputs import *
import matrix
import wf
import potentials as pots
import solver

def abs_sqr(z):
    return np.real(z)**2 + np.imag(z)**2

def tstep(psi_series, H, delta_t, mass,alpha,q,i):
    psi_series[i+1] = solver.prop(psi_series[i],H,0.5*delta_t)
    ## recalculate the H using the predictor GS wavefunction
    H = H_func(psi_series[i+1],np.conjugate(psi_series[i+1]),mass,alpha,q)
    # now calculate the corrector step with the newly calculated H
    psi_series[i+1] = solver.prop(psi_series[i+1],H,delta_t)

def im_tstep(impsi,psi_series, H, delta_t, mass,alpha,q,i,H_func):
    psi_series[i+1] = solver.prop(psi_series[i],H,0.5*delta_t*1.0j)
    impsi[i+1] = solver.prop(impsi[i],H,0.5*delta_t*-1.0j)
    ## recalculate the H using the predictor GS wavefunction
    H = H_func(psi_series[i+1],impsi[i+1],mass,alpha,q)
    # now calculate the corrector step with the newly calculated H
    psi_series[i+1] = solver.prop(psi_series[i],H,delta_t*1.0j)
    impsi[i+1] = solver.prop(impsi[i],H,delta_t*-1.0j)


def animate(i, grid, psi_series, H, delta_t, mass,alpha,q,V):
    plt.cla()
    tstep(psi_series, H, delta_t, mass,alpha,q,i) 
    ## plot each time step
    plt.plot(grid,V)
    plt.plot(grid,np.real(psi_series[i+1]),label='$Re(\psi)$')
    plt.plot(grid,np.imag(psi_series[i+1]),label='$Im(\psi)$')
    plt.plot(grid,np.real(np.conjugate(psi_series[i+1])*psi_series[i+1]),label='$\psi^{*}\psi$')
    plt.title(f't = {np.around(delta_t + delta_t*i,4)} (step {i})')
    plt.ylim([-.1,.2])
    plt.legend()
    plt.show()

def im_animate(i, grid, psi_series, impsi, H, delta_t, mass,alpha,q, V,H_func):
    plt.cla()
    im_tstep(impsi,psi_series, H, delta_t, mass,alpha,q,i,H_func) 
    ## plot each time step
    plt.plot(grid,V)
    plt.plot(grid,np.real(psi_series[i+1]),label='$Re(\psi)$')
    plt.plot(grid,np.imag(psi_series[i+1]),label='$Im(\psi)$')
    plt.plot(grid,np.real(np.conjugate(psi_series[i+1])*psi_series[i+1]),label='$\psi^{*}\psi$')
    plt.title(f't = {np.around(delta_t + delta_t*i,4)} (step {i})')
    plt.ylim([-.1,.2])
    plt.legend()
    plt.show()

psi_func = wf.getPsi_x(0,1,0) ## GS harmonic oscillator.

psiArr = psi_func(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H_func = matrix.get_H_func(name='gaussian')
E_GS,evect_GS = solver.MatrixSolve_SC(H_func,psiArr,psiStarArr)
evect_GS = evect_GS/np.linalg.norm(evect_GS)





## remake the Hamiltonian with GS used in the non-linear interaction

H = H_func(evect_GS,np.conjugate(evect_GS),mass,alpha,q)
print('Numerical GP:', E_GS)
plt.plot(grid,evect_GS)
plt.title('Ground state')
plt.show()


V = pots.V_gaussian(largegrid)

plt.plot(largegrid,V)

#plt.plot(grid,evect_GS**2)

#plt.xlim([-5,5])
#plt.ylim([-1,1])
plt.show()

### Propagation 
psi_series = np.zeros((nt_steps+1,len(largegrid)),dtype=complex)
psi_series[0][:len(psiArr)] = evect_GS.copy()
impsi = np.zeros((nt_steps+1,len(largegrid)),dtype=complex)
impsi[0][:len(psiArr)] = evect_GS.copy()

print('End')

H_func_large = matrix.get_H_func(name='double')
H_large = H_func_large(impsi[0],np.conjugate(impsi[0]),mass,alpha,q)
#if __name__ == "__animate__":
fig = plt.figure()
#ani = FuncAnimation(fig, im_animate, fargs=(grid,psi_series,H,delta_t/2.0,mass,alpha,q,V,), interval=50)
ani = FuncAnimation(fig, im_animate, fargs=(largegrid,psi_series,impsi,H,delta_t/2.0,mass,alpha,q,V,H_func_large), interval=1)
plt.show()
