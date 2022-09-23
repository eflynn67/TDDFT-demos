## Note: hbar = 1 in this solver.
## To run the solver, basically change all of the parameters you want here and
## run main.py in the src file.
###############################################################################
## Self Consistent Solve parameters
niter = 100 # number of self consistent iterations
sigma = 1 #sigma = [0,1]. 0.0 means nothing happens

###############################################################################
## Propagation parameters

prop_order = 6 #Expansion order of the propagator e^(-i \Delta t h(t)).
delta_t = 10**(-4) # time step length rule of thumb is delta_t < 1/N^(1.5)
nt_steps = 100000 #number of time steps

###############################################################################
## Domain properties
#step_size = .2 ## for finite difference schemes
N = 200 # number of collocation points.
lb = -10 # left boundary
rb = 10 # right boundary


## Interaction parameters
mass = 1.0
alpha = 0.5 # interaction strength for quartic and HO potentials (if being used)
q = 0.5# interaction strength for |psi|^2 term in GP Hamiltonian

## Interaction parameters for gaussian potential taken from Barrier penetration
## paper Levit, Negele, and Patiel (1980) (if being used.)
V_1 = -1.489
V_2 = 0.4
V_3 = 0.5
gamma_1 = 2.0
gamma_2 = 10.0