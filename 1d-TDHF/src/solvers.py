import numpy as np
from numba import jit
from init import *

@jit(nopython=True)
def solve_wf_Numerov(psi,g_array,init_side):
    nbox = len(g_array)
    if init_side == 'left':
        psi[0] = 0
        psi[1] = 10**-3
        for i in range(2,nbox):
            a1 = 1.0 + ((step_size**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*step_size**2)/12)*g_array[i-1])
            a3 = -(1+ ((step_size**2)/12)*g_array[i-2])
            psi[i] = (a2*psi[i-1] + a3*psi[i-2])/a1
        norm = 0.0
        for i in range(0,nbox):
            norm += psi[i]**2
        norm = np.sqrt(norm)
        psi = psi/norm
        return psi
    elif init_side =='right':
        psi = psi[::-1]
        psi[0] = 0
        psi[1] = 10**-3
        for i in np.arange(2,nbox,1):
            a1 = 1.0 + ((step_size**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*step_size**2)/12)*g_array[i-1])
            a3 = -(1+ ((step_size**2)/12)*g_array[i-2])
            psi[i] = (a2*psi[i-1] + a3*psi[i-2])/a1
        norm = 0.0
        for i in range(0,nbox):
            norm += psi[i]**2
        norm = np.sqrt(norm)
        return psi[::-1]/norm
    else:
        print('Invalid Wavefunction Initialization')

@jit(nopython=True)
def solve_E_Numerov(psi_init,E0,dE,V):
    '''
    Solves the ODE y''(x) + g(x,E)y(x) = 0 where g(x,E) = E - V(x). E is the eigenvalue of the diff operator.
    The function calls solve_wf_Numerov to solve the ODE given an array for the function g(x,E). To make sure
    the BCs are satisfied, we iterate the eigenvalue E such that psi satisfies the BC. This is done by 

    Parameters
    ----------
    psi_init : nd_array
        Guess for the initial wavfunction
    E0 : float
        Guess for initial energy.
    dE : float
        step size for incrementing energy.
    V : nd array
        the "potential" term evaluated everywhere in the domain.

    Returns
    -------
    E : float
        converged energy.
   psi: nd array
        convered wavefunction.

    '''
    # Construct g-function
    nbox = len(V)
    g_array = E0 - V
    
    psi_l_init = psi_init.copy()
    psi_l = solve_wf_Numerov(psi_l_init,g_array,init_side='left')
    
    
    P1 = psi_l[nbox-1] # grab right side boundary 
    dE_init = dE
    El = E0 + dE
    
    counter = 0
    ### evolve the left initilized one.
    while abs(dE) > 10**-12:
    #for g in range(0,2):
        g_array = El - V
        psi_l = solve_wf_Numerov(psi_l,g_array,init_side='left')
        P2 = psi_l[nbox -1] # grab right side boundary value again
        if P1*P2 < 0:
            dE = - dE/2.0
        El = El + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
    print(f'Left Converged in {counter} Iterations.')
    print('left ',El)
    
    ### Reinitialize the wavefunctions and arrays to solve for the right side initialized wavefunctions
    g_array = E0 - V
    
    psi_r_init = psi_init.copy()
    psi_r = solve_wf_Numerov(psi_r_init,g_array,init_side='right')
    
    P1 = psi_r[0] # grab left side boundary 
    dE = dE_init #reinit the dE iterator
    Er = E0 + dE
    
    counter = 0
    while abs(dE) > 10**-12:
        g_array = Er - V
        psi_r = solve_wf_Numerov(psi_r,g_array,init_side='right')
        P2 = psi_r[0] # grab left side boundary value again
        if P1*P2 < 0:
            dE = - dE/2.0
        Er = Er + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
    # Now merge the right and left solutions together.
    print(f'Right Converged in {counter} Iterations.')
    print('right ',Er)
    if abs(El-Er) < 10**-15:
        E = El
    else: 
        raise Exception('Right and Left Energies dont match. One of theme did not converge. Stopping Solve')
    psi = np.concatenate((psi_l[:int((nbox-1)/2)],psi_r[int((nbox-1)/2):]))
    norm = 0.0
    for i in range(0,nbox):
        norm += psi[i]**2
    norm = np.sqrt(norm)
    psi = psi/norm
    return E,psi

def MatrixSolve():
    energy = None
    psi_array = None
    rho = None
    return energy, psi_array,rho

def ImgTimeSolve():
    energy = None
    psi_array = None
    rho = None
    return energy, psi_array,rho