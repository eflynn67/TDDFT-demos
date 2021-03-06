import numpy as np
from scipy.special import genlaguerre
from scipy.special import assoc_laguerre
from init import *


'''
NOTE: HYDROGEN WAVEFUNCTIONS NOT WORKING
'''
def get_wfHO_radial(n,l):
    '''
    Parameters
    ----------
    n : Integer
        principle quantum number
    l : Integer
        orbital angular momentum quantum number.

    Returns
    -------
    psi : function
        radial wavefunction as a function of r.

    '''
    lag = genlaguerre(n,l+.5)
    nu = .5
    def psi(r):
        result = r**(l) * lag(2*nu*r) * (2 * nu * r**(2)) * np.exp(-nu*r**(2))
        return(result)
    return psi
def get_WfHydrogen_radial(n,l):
    '''
    Taken from Zettili pg 359. Distance scale is relative to charge radius of the proton.
    This means we take a_0 -> R_p

    Returns
    -------
    Radial part of the hydogen wavefunction.

    '''
    def psi(r):
        R_p = .831 # in fm 
        N = (2/(n*R_p))**(3/2) * np.sqrt(np.math.factorial((n - l - 1))/(2*n*np.math.factorial((n+l))**3))
        func = N* ((2*r/(n*R_p))**l) * np.exp(-r/(n*R_p))*assoc_laguerre(r,n-l-1,2*l +1)
        return func
    return psi

def initWfs(name='HO'):
    '''
    Function initializes wavefunctions for proton and neutron according to shell model

    Parameters
    ----------
    N : Integer
        Number of Neutrons.
    Z : Integer
        Number of Protons.
    name : string, optional
        Specify type of initial wavefunction. The default is 'HO'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    psi : TYPE
        DESCRIPTION.

    '''
    if name == 'HO':
        psi_array = np.zeros((2,nmax+1,lmax+1,len(spin),len(grid)))
        energies_array = np.zeros((2,nmax+1,lmax+1,len(spin),1))
        l_lim = 0
        for q in range(2):
            for n in range(nmax+1):
                for l in range(lmax+1):
                    jArr = np.zeros(2)
                    for s in range(len(spin)):
                        #print(q,n,l,s)
                        if l == 0: 
                            j = .5
                            jArr[s] = j
                        else:
                            j = l + spin[s]
                    #jArr = np.unique(jArr)
                    for i,j in enumerate(jArr):    
                        psi_func = get_wfHO_radial(n,j) 
                        eval_psi = psi_func(grid)
                        norm = 0.0
                        for k in range(len(grid)):
                            norm +=  eval_psi[k]**2 *step_size
                        psi_array[q,n,l,i] = eval_psi/np.sqrt(norm)#/np.linalg.norm(eval_psi)
        return psi_array,energies_array
    elif name == 'hydrogen':
        psi_array = np.zeros((2,nmax+1,lmax+1,len(spin),len(grid)))
        energies_array = np.zeros((2,nmax+1,lmax+1,len(spin),1))
        for q in range(2):
            for n in range(nmax+1):
                for l in range(lmax+1):
                    jArr = np.zeros(2)
                    for s in range(len(spin)):
                        print(q,n,l,s)
                        if l == 0: 
                            j = .5
                            jArr[s] = j
                        else:
                            j = l + spin[s]
                    #jArr = np.unique(jArr)
                    for i,j in enumerate(jArr):    
                        psi_func = get_WfHydrogen_radial(n+2, l)
                        eval_psi = psi_func(grid)
                        norm = 0.0
                        for k in range(len(grid)):
                            norm +=  eval_psi[k]**2 *step_size
                        psi_array[q,n,l,i] = eval_psi/np.sqrt(norm)#/np.linalg.norm(eval_psi)
        return psi_array,energies_array
    elif name=='test':
        psi_array = np.zeros((2,nmax,lmax+1,len(spin),len(grid)))
        energies_array = np.zeros((2,nmax,lmax+1,len(spin),1))
        return psi_array,energies_array
    else:
        raise ValueError('Only available wavfunctions are radial HO and hydrogen')
    return None

