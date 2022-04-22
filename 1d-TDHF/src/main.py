from init import *
import densities 
import fields 
import solvers 
import wf
import functionals
import utilities
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


### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])


### Intialize the mean field Hamiltonian h with initial wavefunctions 
hArr = np.zeros(len(grid)) # the hamiltonian
hArr = functionals.h_BKN(rhoArr[0])
## Construct initial g_array. Every wavefunction will have one of these functions
## associated with it. 
fArr = np.zeros((2,nmax+1,lmax+1,len(spin),len(grid)))
### 
D = solvers.getNumerov_matrix()
#E = -10
#dE = .5

for nter in range(0,nIter):
    for q in range(0,2):
        for n in range(nmax+1):
            for l in range(lmax+1):
                for s in range(len(spin)):
                    j = l + spin[s]
                    print(q,n,l,s,j)

                    if q == 0:
                        fArr[q][n][l][s] = .5*grid**2 + hArr + V_c/step_size + fields.centriforceArr(l)*hb2m0 #(hArr + V_c) 
                    else: 
                        fArr[q][n][l][s] = .5*grid**2  +hArr  + fields.centriforceArr(l)*hb2m0#-hArr/hb2m0 - fields.centriforceArr(l)
                    
                    V_matrix = np.diag(fArr[q][n][l][s])
                    H = -hb2m0*D + V_matrix
                    psi = psi_array[q][n][l][s]
                    #energies[q][n][l][s],psi_array[q][n][l][s] = solvers.solve_Numerov(psi,E,dE,fArr[q][n][l][s])
                    energies[q][n][l][s],psi_array[q][n][l][s] = solvers.MatrixNumerovSolve(H)
                    energies[q][n][l][s] = energies[q][n][l][s]
                    print('Converted Energy: ', energies[q][n][l][s][0])
                    
                    ## recalculate rho with new wavegfunction
                    rhoArr = densities.rho(psi_array)
                    V_c = fields.coulombArr(rhoArr[1])
                    hArr = functionals.h_BKN(rhoArr[0])    
                    plt.plot(grid[1:],rhoArr[0][1:])
                    plt.xlabel('r')
                    plt.ylabel(r'$\rho$')
                    plt.title(f'Iteration: {nter}')
                    plt.show()
                    #plt.plot(grid,psi_array[q][n][l][s])
                    #plt.xlabel('r')
                    #plt.ylabel(r'$psi$')
                    #plt.title(f'{nter}')
                    #plt.show() 
                    
    
    #rhoArr = densities.rho(psi_array)
    #plt.plot(grid[:],rhoArr[0][:]/np.linalg.norm(rhoArr[0]))
    #plt.xlabel('r')
    #plt.ylabel(r'$\rho$')
    #plt.title(f'Iteration: {nter}')
    #plt.show()
    npro,nneu = utilities.getNZ(rhoArr)
    print(f' Integrated: N = {nneu}, Z = {npro}')
    Rp,Rn,Rch = utilities.getRadi(rhoArr)
    print(f'Neutron Radius: {Rn}')
    print(f'Proton Radius: {Rp}')
    print(f'Charge Radius: {Rch}')
      
    
    plt.plot(grid,hArr)
    plt.xlabel('r')
    plt.ylabel(r'$h$')
    plt.title(f'{nter}')
    plt.show() 
