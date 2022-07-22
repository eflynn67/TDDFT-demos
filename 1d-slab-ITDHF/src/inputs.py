N = 8
Z = 8
interaction = 'bkn'
coulomb = False
initial_wf = 'HO'
lmax = 1  # max l value for nucleus
nmax = 1 # max n value for nucleus. starts from n=1 which is taken to be the ground state.
nIter = 2 # number of self consistent iterations for the static solver

l_0 = 1.85 # in units of fm
rho_0 = 0.16 # in fm^-3
E_0 = 6.03 # in Mev
t_0 = 1.09*10**-22 # in seconds
## domain properties
lb = -10*l_0 # left boundary
rb = 10*l_0 # right boundary
step_size = .1

## Interaction parameters
V_1 = -1.489
V_2 = 0.4
V_3 = .5
gamma_1 = 2.0
gamma_2 = 10.0