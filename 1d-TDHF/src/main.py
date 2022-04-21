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
plt.plot(grid[:600],rhoArr[1][:600],label='rho init')
plt.legend()
plt.show()
Rn = 0.0 
Rp = 0.0
for i in range(len(grid)):
    Rn += (rhoArr[2][i]*grid[i]**4)*step_size
    Rp += (rhoArr[1][i]*grid[i]**4)*step_size
Rn = np.sqrt(4.0*np.pi*Rn)
Rp = np.sqrt(4.0*np.pi*Rp)
Rch = Rp + 0.8**2
print(f'Neutron Radius: {Rn}')
print(f'Proton Radius: {Rp}')
print(f'Charge Radius: {Rch}')
#print('rhoArr shape: ',rhoArr[0].shape)

### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])


### Intialize the mean field Hamiltonian h with initial wavefunctions 
hArr = np.zeros(len(grid)) # the hamiltonian
hArr = functionals.h_BKN(rhoArr[0])
plt.plot(grid,hArr,label='hArr init')
plt.legend()
plt.show()
## Construct initial g_array. Every wavefunction will have one of these functions
## associated with it. 
fArr = np.zeros((2,nmax,lmax+1,len(spin),len(grid)))
### 
E_array = []
D = solvers.getNumerov_matrix()
#E = E_guess
#dE = .5
A = 0.0
for i in range(1,len(grid)-1):
   A += 4.0*np.pi*rhoArr[0][i]*grid[i]**2 
print(f'A = {A}')
for nter in range(0,nIter):
    for q in range(0,2):
        for n in range(nmax):
            for l in range(lmax+1):
                for s in range(len(spin)):
                    j = l + spin[s]
                    ## fill gArr with initial values for each nucleus. We are using units where m_p = m_n = 1.
                    if q == 0:
                        fArr[q][n][l][s] = (hArr + V_c) - fields.centriforceArr(l)*hb2m0 #
                    else: 
                        fArr[q][n][l][s] = hArr - fields.centriforceArr(l)*hb2m0
                    #plt.plot(grid,fArr[q][n][l][s],label='f function')
                    #plt.legend()
                    #plt.show()
                    V_matrix = np.diag(fArr[q][n][l][s])
                    H = D + V_matrix
                    #E_0,psi_array[q][n][l][s] = solvers.solve_Numerov(psi_array[q][n][l][s],E,dE,fArr[q][n][l][s])
                    energies[q][n][l][s],psi_array[q][n][l][s] = solvers.MatrixNumerovSolve(H)
                    energies[q][n][l][s] = -1*energies[q][n][l][s]*hb2m0
                    
                    
                    
                    print(energies[q][n][l][s][0])
                    #plt.plot(grid[:600],psi_array[q][n][l][s][:600])
                    #plt.xlabel('r')
                    #plt.ylabel(r'$\psi(r)$')
                    #plt.title(f'{nter},{q},{n},{l},{s}')
                    #plt.show()
                    ### recalculate  rho with new wavefunction.
    
    rhoArr = densities.rho(psi_array)
    plt.plot(grid[1:],rhoArr[0][1:])
    plt.xlabel('r')
    plt.ylabel(r'$\rho$')
    plt.title(f'Iteration: {nter}')
    plt.show()
    nneu = 0.0
    npro = 0.0
    for i in range(1,len(grid)-1):
       nneu += 4.0*np.pi*rhoArr[2][i]*grid[i]**2 
       npro += 4.0*np.pi*rhoArr[1][i]*grid[i]**2 
    print(f'N = {nneu}, Z = {npro}')
    ### Calculate radi
    Rn = 0.0 
    Rp = 0.0
    for i in range(len(grid)):
        Rn += (rhoArr[2][i]*grid[i]**4)*step_size
        Rp += (rhoArr[1][i]*grid[i]**4)*step_size
    Rn = np.sqrt(4.0*np.pi*Rn/nneu)
    Rp = np.sqrt(4.0*np.pi*Rp/nneu)
    Rch = Rp + 0.8**2
    print(f'Neutron Radius: {Rn}')
    print(f'Proton Radius: {Rp}')
    print(f'Charge Radius: {Rch}')
    ### recalculate the coulomb  the coulomb field
    V_c = fields.coulombArr(rhoArr[1])
    ### recalculate hamiltonian with new densities
    hArr = functionals.h_BKN(rhoArr[0])         

    #plt.plot(grid,V_c)
    #plt.xlabel('r')
    #plt.ylabel(r'$Vc$')
    #plt.title(f'{nter}')
    #plt.show()    
    
    #plt.plot(grid,hArr)
    #plt.xlabel('r')
    #plt.ylabel(r'$h$')
    #plt.title(f'{nter}')
    #plt.show() 
