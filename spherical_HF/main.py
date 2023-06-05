import numpy as np
import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import sys

import matrix
import potentials
import wf_init

# import 1d solver
sys.path.insert(0, './solvers')
import utilities
import spec
import solvers

'''
This is the main script that runs a pseudo-spectral Chebyshev routine to solve 
the 1-d spherical TDHF equations.Note that in the Chebyshev basis, the problem 
needs to be mapped to the interval [-1,1] and then inverse mapped back to the original domain.
'''
###############################################################################
##### model parameters
Z_pro = 20
N_neu = 20
hb2m0 = 20.735530 # expression for hbar^2 /2mp
coulomb = False # option for to turn on coulomb force on proton

s = .5 #spin of single particle.



Vls = 22 - 14*(N_neu - Z_pro)/(N_neu + Z_pro)# spin orbit strength from Bohr and Mottelson
V0 = -51 + 33*(N_neu - Z_pro)/(N_neu + Z_pro) # MeV WS strength From Bohr and Mottelson

r0 = 1.27 # fm , radius parameter
a = .67 # fm , WS parameter
R = r0*(N_neu + Z_pro)**(1/3) # fm , WS parameter

r_cutoff = .84 # approximate Rp (proton radius) in fm
e2 = 1.4399784 # e^2 charge in MeV fm

kappa = 1.0
###############################################################################


###############################################################################
##### Solver parameters
lb = 10**(-3)## left boundary of box in fm
rb = 15 ## right boundary of box in fm
N = 301 ## number of collocation points
currentInterval = (-1,1) # interval the derivatives are defined on.
targetInterval = (lb,rb) ## interval on real space to solve on

params = {'V0': V0,'R':R,'r0':r0,'a':a,'e2':e2,'r_cutoff': r_cutoff,'Vls': Vls,\
          'Z_pro':Z_pro,'N_neu':N_neu,'hb2m0':hb2m0,'kappa':kappa,'N':N}
    
    
###############################################################################
# Define Collocation points
###############################################################################
# Define chebyshev points in the interval [-1,1]
GL_func = spec.GaussLobatto()
CPnts,weights = GL_func.chebyshev(N)


# Define derivative matrix at the new Cpnts
beta = 0.8
coord_trans = spec.coord_transforms(currentInterval,targetInterval)
#transform = spec.coord_transforms(interval=targetInterval)
CPnts_shift,dgdy_arc = coord_trans.arcTransform(CPnts,beta)

# Define derivative matrix at the new Cpnts
A = np.diag(1.0/dgdy_arc)

## First transform the interval of points [-1,1] to the physical interval

CPnts_mapped,dgdy_affine  = coord_trans.inv_affine(CPnts_shift)
# define derivative matrix on the mapped interval
D_1 = np.matmul(A,spec.DerMatrix().getCheb(CPnts))/dgdy_affine

# define integration weights
int_weights = spec.GaussLobatto().getDx(CPnts_shift)*dgdy_affine

###############################################################################

#H_func = matrix.spec_H_func(N, CPnts_mapped,D_1,params)

###############################################################################
# Set up initial wavefunctions
###############################################################################

sp_states = wf_init.woodSaxon(CPnts_mapped,D_1,params)

