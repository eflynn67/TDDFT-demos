import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.integrate import solve_bvp
from scipy import special

'''
Contains: all the solvers needed to solve 1d single particle self consistent 
problems
'''


###############################################################################
'''

Contents: -1d self consistent solver using pseudospectal method.
          -1d unitary time dependent solver
              - in polynomial basis expansion
              - in chebyshev basis expansion
          - system of ODEs BVP solver using RK4
'''
###############################################################################
def MatrixSolve_SC_hermitian(H_func,psiArr,psiStarArr,int_weights,params):
    niter = params['ninter']
    sigma = params['sigma']
    seriesShape = np.concatenate([[niter],psiArr.shape])
    psiSeries = np.zeros(seriesShape,dtype=complex)
    psiStarSeries = np.zeros(seriesShape,dtype=complex)

    #sigmaSeries = np.zeros(niter) #Array to contain the mixing parameter
    #sigmaSeries[0] = sigma
    #sigmaSeries[1] = sigma
    #rSeries = np.zeros(seriesShape) # residual array

    H = H_func(psiArr,psiStarArr,params)

    evals,evects = sci.linalg.eig(H)
    idx = evals.argsort()
    evals = evals[idx]
    evects = evects[:,idx]
    evects = evects.T

    psi = evects[0]
    psi_norm = utilities.normalize(psi**2,int_weights)
    #psi_norm = wf.normalize(psi**2,int_weights)
    psi = psi/psi_norm
    psi = np.concatenate([[0],psi,[0]])
    psiStar = np.conjugate(psi)/psi_norm
    psiSeries[0] = psi
    psiStarSeries[0] = psiStar

    for l in np.arange(1,niter):
        H = H_func(psiSeries[l-1],psiStarSeries[l-1],params)
        evals,evects = sci.linalg.eig(H)
        idx = evals.argsort()
        evals = evals[idx]
        evects = evects[:,idx]
        evects = evects.T

        psi = evects[0]
        psi = np.concatenate([[0],psi,[0]])
        psi_norm = utilities.normalize(np.abs(psi),int_weights)
        #psi_norm = wf.normalize(np.abs(psi),int_weights)
        psi = psi/psi_norm
        #psiSeries[l] = psi

        psiStar = np.conjugate(psi)/psi_norm
        #psiStarSeries[l] = np.conjugate(psiStar)

        psiSeries[l] = sigma*psi + (1.0-sigma)*psiSeries[l-1]
        psiStarSeries[l] = sigma*psiStar + (1.0-sigma)*psiStarSeries[l-1]
        norm = utilities.normalize(psiSeries[l],int_weights)
        #norm = wf.normalize(psiSeries[l],int_weights)
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
        print(f'GS Energy {l}: {evals[:2]}')
        #norm = wf.normalize(psiSeries[-1]**2,int_weights)
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
     norm = 1.0/utilities.normalize(np.abs(dpsi),weights)
#    norm = 1.0/wf.normalize(np.abs(dpsi),weights)
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


###############################################################################

## This section contains the 1d BVP solver for classical equations of motion

###############################################################################



def bvp(t,x,gradV,params):
    # x is a vector of all the independent variables
    x1, x2 = x
    dx1dt = x2
    dx2dt = gradV(x1,params)*np.exp(-2*1j*(params['theta']))
    return [dx1dt,dx2dt]
def bc(xa,xb,xi,xf):
    #note this is the value of the full sol x at the boundaries
    x1a, x2a = xa  # notion is solution of x1 at the a boundary (x1a), solution of x2 at the a boundary (x2a)
    x1b, x2b = xb
    return([x1a-xi,x1b-xf])






