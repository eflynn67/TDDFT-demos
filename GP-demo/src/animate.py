import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter

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
    #H = H_func(psi_series[i],impsi[i],mass,alpha,q)
    H = H_func(psi_series[i],impsi[i],mass,alpha,q,largegrid,shift=shift)
    
    if(derv==None or derv==True):
        impsi[i+1] = solver.prop(impsi[i],H,0.5*delta_t*-1.0j)
        psi_series[i+1] = solver.prop(psi_series[i],H,0.5*delta_t*1.0j)
    else:
        impsi[i+1] = solver.splineprop(impsi[i],H,0.5*delta_t*-1.0j,evect_GS_spline)
        psi_series[i+1] = solver.splineprop(psi_series[i],H,0.5*delta_t*1.0j,evect_GS_spline)
    ## recalculate the H using the predictor GS wavefunction
    #H = H_func(psi_series[i+1],impsi[i+1],mass,alpha,q,)
    H = H_func(psi_series[i+1],impsi[i+1],mass,alpha,q,largegrid,shift=shift)
    # now calculate the corrector step with the newly calculated H
    if(derv==None or derv==True):
        psi_series[i+1] = solver.prop(psi_series[i],H,delta_t*1.0j)
        impsi[i+1] = solver.prop(impsi[i],H,delta_t*-1.0j)
    else:
        psi_series[i+1] = solver.splineprop(psi_series[i],H,delta_t*1.0j,evect_GS_spline)
        impsi[i+1] = solver.splineprop(impsi[i],H,delta_t*-1.0j,evect_GS_spline)


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
    plt.plot(grid,V+np.real(pots.V_rho(psi_series[i+1],impsi[i+1],q)))
    plt.plot(grid,np.real(psi_series[i+1]),label='$Re(\psi)$')
    #plt.plot(grid,np.imag(psi_series[i+1]),label='$Im(\psi)$')
    #plt.plot(grid,np.real(impsi[i+1])+0.5,label='$Re(\psi_i)$')
    #plt.plot(grid,np.imag(impsi[i+1])+0.5,label='$Im(\psi_i)$')
    plt.plot(grid,np.real(np.conjugate(psi_series[i+1])*psi_series[i+1]),label='$\psi^{*}\psi$',color='r')
    realrho=np.real(np.conjugate(psi_series[i+1])*psi_series[i+1])
    imrho=np.real(np.conjugate(impsi[i+1])*impsi[i+1])
    psimax=np.argmax(realrho)
    immax=np.argmax(imrho)
    plt.plot(grid,np.real(np.conjugate(impsi[i+1])*impsi[i+1])+0.2,label='$\psi_i^{*}\psi_i$',color='g')
    plt.vlines(grid[immax],ymin=-10,ymax=10,label='Conjugate Maximum',color='g',ls='--')
    plt.vlines(grid[psimax],ymin=-10,ymax=10,label='Psi Maximum',color='r',ls='--')
    plt.title(f't = {np.around(delta_t + delta_t*i,4)} (step {i})')
    plt.ylim([-.5,.3])
    plt.legend()
    plt.show()

psi_func = wf.getPsi_x(0,1,0) ## GS harmonic oscillator.

psiArr = psi_func(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H_func = matrix.get_H_func(name='gaussian')
E_GS,tevect_GS = solver.MatrixSolve_SC(H_func,psiArr,psiStarArr)
tevect_GS = tevect_GS/np.linalg.norm(tevect_GS)


evect_GS_spline = interpolate.splrep(grid, tevect_GS, s=0, k=5)
evect_GS = interpolate.splev(grid, evect_GS_spline, der=0)

derv=True
## remake the Hamiltonian with GS used in the non-linear interaction

H = H_func(evect_GS,np.conjugate(evect_GS),mass,alpha,q)
print('Numerical GP:', E_GS)
plt.plot(grid,tevect_GS)
plt.plot(grid,evect_GS)
plt.title('Ground state')
plt.show()

shift=5
V = pots.V_gaussian(largegrid,shift=-1)+pots.V_HO(largegrid,alpha,4) + pots.V_gaussian(largegrid,shift=shift,V_1=-2)
#V = pots.V_gaussian(largegrid)
plt.plot(largegrid,V)

#plt.plot(grid,evect_GS**2)

#plt.xlim([-5,5])
#plt.ylim([-1,1])
plt.show()

### Propagation
boost=-12.0
evect_GS = np.exp(1.0j*boost*grid)*evect_GS
psi_series = np.zeros((nt_steps+1,len(largegrid)),dtype=complex)
psi_series[0][:len(psiArr)] = evect_GS.copy()
impsi = np.zeros((nt_steps+1,len(largegrid)),dtype=complex)
impsi[0][:len(psiArr)] = evect_GS.copy()

print('End')

H_func_large = matrix.get_H_func(name='double')
#H_large = H_func_large(impsi[0],np.conjugate(impsi[0]),mass,alpha,q)
#H_large = H_func_large(impsi[0],np.conjugate(impsi[0]),mass,alpha,q,largegrid,shift=shift)
H_large = H_func_large(impsi[0],np.conjugate(impsi[0]),mass,alpha,q,largegrid,shift=shift,derv=derv)
#if __name__ == "__animate__":
fig = plt.figure()
#ani = FuncAnimation(fig, im_animate, fargs=(grid,psi_series,H,delta_t/2.0,mass,alpha,q,V,), interval=50)
ani = FuncAnimation(fig, im_animate, fargs=(largegrid,psi_series,impsi,H_large,delta_t/2.0,mass,alpha,q,V,H_func_large), interval=1,repeat=False,save_count=nt_steps)
f = r"0005dt.mp4" 
#writervideo = FFMpegWriter(fps=60) 
#ani.save(f, writer=writervideo)
plt.show()
