import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import math
from scipy import special
import matrix

def MatrixSolve_SC(H_func,psiArr,psiStarArr):
    H = H_func(psiArr,psiStarArr,mass,alpha,q)

    energies,evects = sci.linalg.eig(H,subset_by_index = [0,0])
    evects = evects.T

    norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
    conj_evect = np.conjugate(evects[0])

    for l in range(niter):
        H = H_func(evects[0],conj_evect,mass,alpha,q)
        #energies,evects = sci.linalg.eigh(H,subset_by_index = [0,0])
        energies,evects = sci.linalg.eig(H)
        idx = evals.argsort()
        evals = evals[idx]
        evects = evects[:,idx]
        evects = evects.T
        norm = 1.0/np.linalg.norm(evects[0])
        conj_evect = np.conjugate(evects[0])
        print(f'Energies {l}: {energies[0]}')
    norm = 1/(np.linalg.norm(evects[0]))
    evects[0] = evects[0]*norm
    return energies[0],evects[0]

def prop_cheb(psi,H,dt,prop_order):
    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += (- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        elif k ==1 :
            dpsi += 2.0*(- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = 2.0*np.matmul(H_normalized,dpsi_buffer[k-1]) - dpsi_buffer[k-2]
            dpsi += (2.0*(- 1.0j)**k) * special.jv(k,tau)*dpsi_buffer[k]

    norm = 1.0/np.linalg.norm(dpsi)
    dpsi = dpsi*norm
    return dpsi

def prop(psi,H,dt,prop_order):
    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += ((- 1.0j)**k)*dpsi_buffer[k]
        elif k ==1 :
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = np.matmul(H_normalized,dpsi_buffer[k-1]) - dpsi_buffer[k-2]
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]/math.factorial(k)
    norm = 1.0/np.linalg.norm(dpsi)
    dpsi = dpsi*norm
    return dpsi


