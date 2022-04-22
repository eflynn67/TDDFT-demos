N = 8
Z = 8
interaction = 'bkn'
coulomb = True
initial_wf = 'HO'
lmax = 1  # max l value for nucleus
nmax = 0 # max n value for nucleus. starts from n=1 which is taken to be the ground state.
nIter = 200
#E_guess = -50 # initial HF energy in MeV


## domain properties
lb = 10**-20 # left boundary
rb = 20 # right boundary
step_size = .1

## Interaction parameters
a = 0.45979 # length parameter for yukawa potential (fm)
aV0 = -166.9239 # strength of yukwawa potential (MeV)

t1 = -497.726 #MeVfm^3
t3 = 17270 #MeVfm^6