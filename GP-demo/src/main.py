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

def animate(i, x=[], y=[]):
    plt.cla()
    x.append(i)
    y.append(random.randint(0, 10))
    plt.plot(x, y)

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


V = pots.V_gaussian(grid)
plt.plot(grid,V)
plt.plot(grid,evect_GS**2)
plt.xlim([-5,5])
plt.ylim([-1,1])
plt.show()

### Propagation 
psi_series = np.zeros((nt_steps+1,len(psiArr)),dtype=complex)
psi_series[0] = evect_GS.copy()

for j in range(nt_steps):
    # predictor step
    psi_series[j+1] = solver.prop(psi_series[j],H,0.5*delta_t)
    ## recalculate the H using the predictor GS wavefunction
    H = H_func(psi_series[j+1],np.conjugate(psi_series[j+1]),mass,alpha,q)
    # now calculate the corrector step with the newly calculated H
    psi_series[j+1] = solver.prop(psi_series[j+1],H,delta_t)
    
    ## plot each time step
    plt.plot(grid,np.real(psi_series[j+1]),label='$Re(\psi)$')
    plt.plot(grid,np.imag(psi_series[j+1]),label='$Im(\psi)$')
    plt.plot(grid,np.real(np.conjugate(psi_series[j+1])*psi_series[j+1]),label='$\psi^{*}\psi$')
    plt.title(f't = {np.around(delta_t + delta_t*j,4)} (step {j})')
    plt.ylim([-.1,.2])
    plt.legend()
    plt.show()
    
print('End')


if __name__ == "__main__":
    fig = plt.figure()
    ani = FuncAnimation(fig, animate, interval=700)
    plt.show()
