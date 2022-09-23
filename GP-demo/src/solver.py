import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import math
from scipy import special
import matrix
from inputs import *
import wf

def MatrixSolve_SC_hermitian(H_func,psiArr,psiStarArr,int_weights):
    seriesShape = np.concatenate([[niter],psiArr.shape])
    psiSeries = np.zeros(seriesShape,dtype=complex)
    psiStarSeries = np.zeros(seriesShape,dtype=complex)

    #sigmaSeries = np.zeros(niter) #Array to contain the mixing parameter
    #sigmaSeries[0] = sigma
    #sigmaSeries[1] = sigma
    #rSeries = np.zeros(seriesShape) # residual array

    H = H_func(psiArr,psiStarArr,mass,alpha,q)

    evals,evects = sci.linalg.eig(H)
    idx = evals.argsort()
    evals = evals[idx]
    evects = evects[:,idx]
    evects = evects.T

    psi = evects[0]
    psi_norm = wf.normalize(psi**2,int_weights)
    psi = psi/psi_norm
    psi = np.concatenate([[0],psi,[0]])
    psiStar = np.conjugate(psi)/psi_norm
    psiSeries[0] = psi
    psiStarSeries[0] = psiStar

    for l in np.arange(1,niter):
        H = H_func(psiSeries[l-1],psiStarSeries[l-1],mass,alpha,q)
        evals,evects = sci.linalg.eig(H)
        idx = evals.argsort()
        evals = evals[idx]
        evects = evects[:,idx]
        evects = evects.T

        psi = evects[0]
        psi = np.concatenate([[0],psi,[0]])

        psi_norm = wf.normalize(np.abs(psi),int_weights)
        psi = psi/psi_norm
        #psiSeries[l] = psi

        psiStar = np.conjugate(psi)/psi_norm
        #psiStarSeries[l] = np.conjugate(psiStar)

        psiSeries[l] = sigma*psi + (1.0-sigma)*psiSeries[l-1]
        psiStarSeries[l] = sigma*psiStar + (1.0-sigma)*psiStarSeries[l-1]
        norm = wf.normalize(psiSeries[l],int_weights)
        psiSeries[l] = psiSeries[l]/norm
        psiStarSeries[l] = psiStarSeries[l]/norm

        '''
        #mixing parameters give different results.
        #calculate residual
        rSeries[l] = psiSeries[l] - psiSeries[l-1]
        # compute the optimal mixing parameter
        if l > 1:
            r_norm = np.linalg.norm(rSeries[l] -rSeries[l-1])
            #print(r_norm)
            if r_norm < 10**(-4):
                #print('it vanished')
                sigmaSeries[l] = 0.0
            else:
                #print('it did not vanish')

                sigmaSeries[l] = -1.0*np.dot(rSeries[l-1],rSeries[l] -rSeries[l-1])/r_norm
                print(np.dot(rSeries[l-1],rSeries[l] -rSeries[l-1]))
                print(r_norm)
                #print(sigmaSeries[l])
        else: pass
        '''
        #print(f'GS Energy {l}: {evals[:2]}')
        norm = wf.normalize(psiSeries[-1]**2,int_weights)

    return evals[0],psiSeries[-1]/norm

def prop_cheb(psi,H,dt,prop_order,weights):
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

    norm = 1.0/wf.normalize(np.abs(dpsi),weights)
    dpsi = dpsi*norm
    return dpsi

def prop(psi,H,dt,prop_order,weights):
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
    norm = 1.0/wf.normalize(np.abs(dpsi),weights)
    dpsi = dpsi*norm
    return dpsi


