## Note: hbar = 1 in this solver.
## To run the solver, basically change all of the parameters you want here and
## run main.py in the src file.
coulomb = False # not used anywhere lol.
niter = 8 # number of self consistent iterations
prop_order = 6 # expansion order of the propagator e^(-i \Delta t h(t)).

## Domain properties
N = 200 # number of collocation points.
lb = -10 # left boundary
rb = 10 # right boundary
delta_t = 10**(-4) # time step length
nt_steps = 1000000 #number of time steps



## Interaction parameters
mass = 1.0
alpha = 0.5 # interaction strength for quartic and HO potentials (if being used)
q = 1.0 # interaction strength for |psi|^2 term in GP Hamiltonian

## Interaction parameters for gaussian potential taken from Barrier penetration
## paper Levit, Negele, and Patiel (1980) (if being used.)
V_1 = -1.489
V_2 = 0.4
V_3 = 0.5
gamma_1 = 2.0
gamma_2 = 10.0