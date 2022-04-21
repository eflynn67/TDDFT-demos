import numpy as np
from numba import jit
from init import *
import matplotlib.pyplot as plt
#@jit(nopython=True)
def solve_wf_Numerov(psi,g_array,init_side):
    #psi = psi_in.copy()
    nbox = len(g_array)
    if init_side == 'left':
        psi[0] = 0
        psi[1] = 10**-3
        for i in range(2,nbox):
            a1 = 1.0 + ((step_size**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*step_size**2)/12)*g_array[i-1])
            a3 = -(1+ ((step_size**2)/12)*g_array[i-2])
            #print(a1,a2,a3)
            psi[i] = (a2*psi[i-1] + a3*psi[i-2])/a1
        #norm = 0.0
        #for i in range(0,nbox):
            #norm += psi[i]**2
        #norm = np.sqrt(norm)
        psi = psi#/norm
        return psi
    elif init_side =='right':
        #psi = psi[::-1]
        psi[nbox-1] = 0
        psi[nbox-2] = grid[1]
        for i in np.arange(nbox-3,-1,-1):
            a1 = 1.0 + ((step_size**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*step_size**2)/12)*g_array[i+1])
            a3 = -(1+ ((step_size**2)/12)*g_array[i+2])
            psi[i] = (a2*psi[i+1] + a3*psi[i+2])/a1
        #norm = 0.0
        #for i in range(0,nbox):
        #    norm += psi[i]**2
        #norm = np.sqrt(norm)
        return psi#/norm
    else:
        print('Invalid Wavefunction Initialization')

def solve_Numerov(psi_init,E0,dE,V):
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
    njoin = 700
    g_array = E0 + V
    
    psi_l_init = psi_init.copy()
    
    psi_l = solve_wf_Numerov(psi_l_init,g_array,init_side='left')
    #plt.plot(grid,psi_l)
    #plt.show()
    P1 = psi_l[nbox-1] # grab right side boundary 
    dE_init = dE
    El = E0 + dE
    
    counter = 0
    ### evolve the left initilized one.
    while abs(dE) > 10**-12:
    #for p in range(0,200):
        #print(E)
        g_array = El + V
        psi_l = solve_wf_Numerov(psi_l,g_array,init_side='left')
        P2 = psi_l[nbox -1] # grab right side boundary value again
        if P1*P2 < 0:
            dE = - dE/4.0
        El = El + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
        #print('P2', P2)
    #print(f'Left Converged in {counter} Iterations.')
    print('left energy',El)
    
    ### Reinitialize the wavefunctions and arrays to solve for the right side initialized wavefunctions
    g_array = E0 + V
    
    psi_r_init = psi_init.copy()
    psi_r = solve_wf_Numerov(psi_r_init,g_array,init_side='right')
    #plt.plot(grid,psi_r)
    #plt.title('right')
    #plt.show()
    P1 = psi_r[0] # grab left side boundary 
    dE = dE_init #reinit the dE iterator
    Er = E0 + dE
    
    counter = 0
    while abs(dE) > 10**-12:
    #for p in range(0,1):    
        g_array = Er + V
        psi_r = solve_wf_Numerov(psi_r,g_array,init_side='right')
        P2 = psi_r[0] # grab left side boundary value again
        if P1*P2 < 0:
            dE = - dE/4.0
        Er = Er + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
    # Now merge the right and left solutions together.
    #print(f'Right Converged in {counter} Iterations.')
    print('right energy',Er)
    print('max psi val right', max(psi_l))
    if abs(El-Er) < 10**-15:
        E = El
    else: 
        raise Exception('Right and Left Energies dont match. One of theme did not converge. Stopping Solve')
    print(psi_r[nbox-100],psi_r[nbox-11],psi_r[nbox-22],psi_r[nbox-33],psi_r[nbox-44],psi_r[nbox-55])
    #plt.plot(grid,psi_l)
    #plt.plot(grid[1500:nbox-1],psi_r[1500:nbox-1])
    #plt.show()
    
    zero_pad = np.zeros(len(grid))
    psi = np.concatenate((psi_l[:njoin],zero_pad[njoin:]))
    norm = 0.0
    for i in range(0,nbox):
        norm += psi[i]**2
    norm = np.sqrt(norm)
    print(norm)
    psi = psi/norm
    return El,psi/norm

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