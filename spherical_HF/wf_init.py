import numpy as np
import potentials
import matrix
import sys
sys.path.insert(0, './solvers')
import utilities

def woodSaxon(CPnts_mapped,D_1,params):
    N = params['N'] #number of collocation points
    nmax_neu,lmax_neu = utilities.getMax_n_l(params['N_neu']) # max quantum number n to solve for
    nmax_pro,lmax_pro = utilities.getMax_n_l(params['Z_pro'])
    H_func = matrix.spec_H_func(N, CPnts_mapped,D_1,params)
    
    '''
    Vws = potentials.ws(CPnts_mapped,params)
    Vc = potentials.coulomb(CPnts_mapped,params)
    Vcent = potentials.centrifugal(CPnts_mapped,l)
    Vso = potentials.spin_orbit(CPnts_mapped,j,l,params)
    Vtot = params['hb2m0']*Vcent + Vws + Vso
    '''
    
    H = H_func.spherical_ws(j,l,params,coulomb=coulomb,BC=True)
    return

def HO():
    return