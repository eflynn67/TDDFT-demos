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

