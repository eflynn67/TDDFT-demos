from init import *
def V_HO(x,alpha):
    '''
    1-d harmonic Oscillator potential

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator length parameter.

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return alpha*x**2

def V_rho(psi,psiStar,q):
    '''

    Parameters
    ----------
    psi : ndArray
        wavefunction.
    psiStar : TYPE
        conjugate wavefunction
    q : TYPE
        coupling constant.

    Returns
    -------
    result : ndArray
           nonlinear part of the GP equation.

    '''
    #print(type(psiStar))
    #print(type(psi))
    #print(type(q))
    #print(type(q*psiStar*psi))
    return q*psiStar*psi

def V_quartic(x,alpha):
    '''
    1-d harmonic Oscillator potential

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator length parameter.

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return alpha*(x**2 -1)**2

def V_gaussian(x):
    if isinstance(x, np.ndarray) == False:
        x = np.array(x)
    if len(x.shape) >=1:
        V = np.zeros(nbox)
    else:
        V = 0
    V_const = [V_1,V_2]
    gamma_const = [gamma_1,gamma_2]
    for i in range(0,2):
        V += (V_const[i]/(np.sqrt(np.pi)*gamma_const[i]))*np.exp(-(x/gamma_const[i])**2)
    return V