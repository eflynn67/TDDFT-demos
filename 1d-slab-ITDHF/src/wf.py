from init import *
from inputs import *
from scipy import special


## initial wavfunctions
def getHO(n,k):
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
        psi_array = np.zeros((2,nmax+1,len(spin),len(grid)))
        energies_array = np.zeros((2,nmax+1,len(spin),1))
        l_lim = 0
        for q in range(2):
            for n in range(nmax+1):
                for s in range(len(spin)):
                    psi_func = getHO(n,.5)# hardcoding an oscillator length cause i suck.
                    norm = 0.0
                    eval_psi= psi_func(grid)
                    for k in range(len(grid)):
                        norm +=  eval_psi[k]**2 *step_size
                    psi_array[q][n][s] = eval_psi/np.sqrt(norm)
        return psi_array,energies_array