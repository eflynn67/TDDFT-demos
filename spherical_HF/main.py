import numpy as np

import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import matrix
import potentials
import sys
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
# based on N_neu and Z_pro, get the maximum n and l. 
nmax = utilities.getMax_n(N_neu, Z_pro) # max quantum number n to solve for
lmax = utilities.getMax_l(N_neu, Z_pro) # max orbital angular momentum to solve for

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
          'Z':Z_pro,'N':N_neu,'hb2m0':hb2m0,'kappa':kappa}




