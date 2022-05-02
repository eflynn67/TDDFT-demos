N = 2
Z = 2
interaction = 'bkn'
coulomb = True
initial_wf = 'HO'
lmax = 0  # max l value for nucleus
nmax = 0 # max n value for nucleus. starts from n=1 which is taken to be the ground state.
nIter = 2
#E_guess = -50 # initial HF energy in MeV


## domain properties
lb = 10**-25 # left boundary
rb = 20 # right boundary
step_size = .02

## Interaction parameters
a = .45979#0.45979 # length parameter for yukawa potential (fm)
aV0 = -16.69239#-166.9239 # strength of yukwawa potential (MeV)

t0 = -497.726#-497.726 #MeVfm^3
t1 = 271.67
t2 = -138.33
t3 = 17270.0#13757.0 #MeVfm^6