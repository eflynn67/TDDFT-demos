## Note: hbar = 1 in this solver.
## To run the solver, basically change all of the parameters you want here and 
## run main.py in the src file.
coulomb = False # not used anywhere lol.
niter = 10 # number of self consistent iterations
prop_order = 8 # expansion order of the propagator e^(-i \Delta t h(t)).

## Domain properties
lb = -20 # left boundary
rb = 20# right boundary
step_size = .2 #spatial grid spacing
delta_t = .001 # time step length
nt_steps = 5000 #number of time steps 

gridextension = 5


## Interaction parameters
mass = 1.0 
alpha = 0*1e-3 # interaction strength for quartic and HO potentials (if being used)
q = -0.0 # interaction strength for |psi|^2 term in GP Hamiltonian

## Interaction parameters for gaussian potential taken from Barrier penetration
## paper Levit, Negele, and Patiel (1980) (if being used.)
V_1 = -1.489
V_2 = 0.4
V_3 = 0.5
gamma_1 = 2.0
gamma_2 = 10.0