import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

from init import *
from inputs import *
from matrix import construct_H_GP
import potentials 
import solver

def getPsi_x(n,k):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.
    
    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    k : TYPE
        harmonic oscillator parameter (mass*omega)^2 from the potential term \mu^2 \omega^{2} x^{2} in the SE.

    Returns
    -------
    wf : function
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x. 
    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*(k)**(.25)))*np.exp(-x**(2)*np.sqrt(k)/2)*herm(x*k**(.25))
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

psi = getPsi_x(0,1) ## GS harmonic oscillator.

psiArr = psi(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H = construct_H_GP(psiArr,psiStarArr,mass,alpha,q)

E,evects = solver.MatrixSolve_SC(H)
E_HO =  getExactLambda(0,mass,alpha)
print('Exact HO: ',E_HO)
print('Numerical GP:', E)
plt.plot(grid,evects,label=f'GP q = {q}')
plt.plot(grid,psiArr,label=f'HO')
plt.legend()
plt.show()




