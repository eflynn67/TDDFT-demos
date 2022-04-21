from init import *
import densities 
import fields 
import solvers 
import wf
import functionals
import matplotlib.pyplot as plt
### Initialize the single particle wavefunctions
#print(f'N: {N}, Z: {Z}')
#print(f'Initilizaing Wavefunctions with {initial_wf}')
psi_array,energies = wf.initWfs(N,Z,name='HO')
#print('psi_array shape:', psi_array.shape)
#print('shape labels: (q,n,l,s)')

### Initialize the Densities (\rho)
#print(f'Initilizaing Wavefunctions with {initial_wf}')
rhoArr = densities.rho(psi_array)
#print('rhoArr shape: ',rhoArr[0].shape)

### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])


### Intialize the mean field Hamiltonian h with initial wavefunctions 
hArr = np.zeros(len(grid)) # the hamiltonian
hArr = functionals.h_BKN(rhoArr[0])

## Construct initial g_array. Every wavefunction will have one of these functions
## associated with it. 
fArr = np.zeros((2,nmax,lmax+1,len(spin),len(grid)))


E = E_guess
dE = .5
for nter in range(0,3):
    for q in range(0,2):
        for n in range(nmax):
            for l in range(lmax+1):
                for s in range(len(spin)):
                    j = l + spin[s]
                    ## fill gArr with initial values for each nucleus. We are using units where m_p = m_n = 1.
                    if q == 0:
                        fArr[q][n][l][s] = -1*(2.0*hArr + 2.0*V_c)*hb2m0 -l*(l+1)/grid**2 #
                    else: 
                        fArr[q][n][l][s] = -1*2.0*hArr*hb2m0 - l*(l+1)/grid**2
                    #plt.plot(grid,psi_array[q][n][l][s],label='init before solve')
                    #plt.legend()
                    #plt.show()
                    E_0,psi_array[q][n][l][s] = solvers.solve_Numerov(psi_array[q][n][l][s],E,dE,fArr[q][n][l][s])
                    print(E_0/(hb2m0))
                    #plt.plot(grid[:],psi_array[q][n][l][s][:])
                    #plt.xlabel('r')
                    #plt.ylabel(r'$\psi(r)$')
                    #plt.title(f'{nter},{q},{n},{l},{s}')
                    #plt.show()
                    ### recalculate  rho with new wavefunction.
    
    rhoArr = densities.rho(psi_array)
    plt.plot(grid,rhoArr[0])
    plt.xlabel('r')
    plt.ylabel(r'$\rho$')
    plt.title(f'{nter}')
    plt.show()
                    ### recalculate the coulomb  the coulomb field
    V_c = fields.coulombArr(rhoArr[1])
                    ### recalculate hamiltonian with new densities
    hArr = functionals.h_BKN(rhoArr[0])         

    plt.plot(grid,V_c)
    plt.xlabel('r')
    plt.ylabel(r'$Vc$')
    plt.title(f'{nter}')
    plt.show()    
    
    plt.plot(grid,hArr)
    plt.xlabel('r')
    plt.ylabel(r'$h$')
    plt.title(f'{nter}')
    plt.show() 
                    #plt.plot(grid,psi_array[q][n][l][s])
                    #plt.show()
                    #psi_array[q][n][l][s] = psi_test
                    #energies[q][n][l][s] = E_0
                    #print('Numerical E_0: ',E_0)