import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import math
from numba import jit

@jit(nopython = True,fastmath=True)
def normalize(funcArr,weights):
    norm = 0.0
    for i in range(len(weights)-1):
        norm += weights[i]*funcArr[i]
    norm = np.sqrt(norm)
    return norm

def getPsi_x(n,k,mu):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.

    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    k : TYPE
        harmonic oscillator parameter (mass*omega)^2 from the potential term \mu^2 \omega^{2} x^{2} in the SE.
    mu: float
        center of the wavefunction
    Returns
    -------
    wf : function
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x.
    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*(k)**(.25)))*np.exp(-(x-mu)**(2)*np.sqrt(k)/2)*herm((x-mu)*k**(.25))
        return(result)
    return wf
def getExactLambda(n,mass,alpha):
    '''
    Exact eigenvalues of the HO equation. -f''(x) + k x^2 f(x) = 2 m E f(x)
    Lambda = 2 m E
    E = (n + .5) \omega
    \alpha = m^2 \omega^2
    \omega = \sqrt{alpha/m^2}
    Parameters
    ----------
    n : float
        principle quantum number. float integers
    omega : float
        oscillator frequency.

    Returns
    -------
    float
        oscillator energy 2mE.
    '''
    return 2*mass*(.5 + n)*np.sqrt(alpha/mass**2)