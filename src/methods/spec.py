import numpy as np

class GaussLobatto:
    def chebyshev(self,N):
        '''
        Gets the Gauss-Lobatto points of the Mth chebyshev polynomials.
        Formula taken from Spectral Methods in Fluid Dynamics (1988).

        These points are only defined on the interval [-1,1] corresponding
        to the domain of the chebyshev polynomials.
        N: integer
            max number of basis elements
        '''
        GSPnts = np.zeros(N+1)
        for l in range(N+1):
            GSPnts[l] = np.cos(l*np.pi/N)
        return GSPnts

    def fourier(self,N):
        '''
        Gets the Gauss-Lobatto points of the Mth Fourier basis element.
        Formula taken from Spectral Methods in Fluid Dynamics (1988)

        These points are defined on the interval [0,2pi] corresponding to the
        domain of the Fourier basis functions.
        N: integer
            max number of basis elements

        '''
        CPnts = np.zeros(N+1)
        for j in range(N+1):
            CPnts[j] = 2.0*j*np.pi/N
        return CPnts

class DerMatrix:
    '''
    The class contains the derivative matrices for Chebyshev and Fourier
    Collocation. The matricies are defined on the intervals [-1,1] and [0,2pi]
    for Chebyshev and Fourier expansions respectively.
    '''
    def _c_coeff(self,l,N):
        '''
        coefficients needed to define derivative matricies. Taken from
        Spectral Methods in Fluid Dynamics (1988).

        Parameters
        ----------
        l : integer
            DESCRIPTION.
        N : integer
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        '''
        if l == 0 or l == N:
            return 2.0
        else:
            return 1.0
    def getCheb(self,CPnts):
        '''
        Analytic solution taken from Spectral Methods in Fluid Dynamics (1988) pg 69 (or 84 in pdf).
        Assumes you are taking the collocation points at the Gauss-Lobatto points

        i labels the collocation point, j labels C
        Parameters
        ----------
        Cpnts: array
            array of collocation points to evaluate the derivative matrix at
        BC: string
            set the type of boundary conditions you want.
            dirichlet: this returns a matrix of size N - 2. This is because we remove
            two rows and two columns. This is because the dirichlet BCs allows us to move
            the boundary terms in the non-homogenous part

        Returns
        -------
        None.

        '''
        N = len(CPnts) - 1
        D = np.zeros((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                if i == 0 and j == 0:
                    D[i][j] = -(2 * N**2 + 1)/6
                elif i == N  and j == N:
                    D[i][j] = (2 * N**2 + 1)/6
                elif i == j and j <= N and j >= 1 and i <= N  and i >= 1 :
                    D[i][j] = - CPnts[j]/(2*(1 - CPnts[j]**2))
                else:
                    D[i][j] = self._c_coeff(i,N)*(-1)**(i+j) /(self._c_coeff(j,N)*(CPnts[i] - CPnts[j]))
        return D

    def getFourier(self,CPnts):
        '''
        Analytic solution taken from Spectral Methods in Fluid Dynamics (1988) pg 69 (or 84 in pdf).
        Assumes you are taking the collocation points at the Gauss-Lobatto points

        i labels the collocation point, j labels C
        Parameters
        ----------
        Cpnts: array
            array of collocation points to evaluate the derivative matrix at
        BC: string
            set the type of boundary conditions you want.
            dirichlet: this returns a matrix of size N - 2. This is because we remove
            two rows and two columns. This is because the dirichlet BCs allows us to move
            the boundary terms in the non-homogenous part

        Returns
        -------
        None.
        '''
        N = len(CPnts)
        S = np.zeros((N,N),dtype='complex')
        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i][j] = .5*(-1)**(i+j) * 1.0/np.tan((i-j)*np.pi/N)
        return S

class coord_transforms:
    def __init__(self,interval):
        self.interval = interval
        self.a = min(interval)
        self.b = max(interval)
        self.alpha1 = .5*(self.b-self.a)
        self.alpha2 = .5*(self.a+self.b)

    def affine(self,Pnts):
        '''
        performs an affine transformation of the coordinate variables to the
        inteval [-1,1] of the form
        x' = a x + b
        f : [a,b] -> [-1,1]
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform

        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        Pnts = Pnts/self.alpha1  - self.alpha2/self.alpha1
        dgdy = 1.0/self.alpha1

        return Pnts_mapped,dgdy
    def inv_affine(self,Pnts):
        '''
        Performs an inverse affine transformation out of the interval [-1,1].

        x = a x' + b

        f : [-1,1] -> [a,b]
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform

        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        Pnts_mapped = Pnts*self.alpha1 + self.alpha2
        dgdy = self.alpha1
        return Pnts_mapped,dgdy
    def arctanh(self,Pnts,L):
        '''
        Performs a arctanh transformation of coordinates on the interval [-1,1].
        This is necessary if we have boundary conditions at x = +/- \inf.

        f: [-1,1] -> (-inf,inf)
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform
        L: float
            length scale of the mapping
        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        Pnts_mapped = L* np.arctanh(Pnts)
        dgdy = L/(1 + Pnts**2)
        return Pnts_mapped,dgdy
    def cotTransform(self,Pnts,L):
        '''
        Coordinate transform that maps points from [0,2pi]. Taken from Spectral Methods
        in Fluid Dynamics (1988)

        this function maps f: [0,2pi]-> (-inf,inf)
        Cpnts must be in the interval [0,2pi].
        Parameters
        ----------
        Cpnts : ndarray
            grid points to transform.
        L : float
            parameter to control the grid spacing scale.

        Returns
        -------
        result : ndarray
            transformed grid points

        '''
        Pnts_mapped = -L* 1.0/np.tan(.5*Pnts)
        dgdy = .5*L/(np.sin(.5*Pnts)**2)
        return Pnts_mapped,dgdy
    def arcTransform(self,Pnts,beta):
        '''
        Transformation for Chebyshev Gauss_Lobatto points. Transformation taken from
        Kosloff and Tal-Ezer (1991)
        this function maps f: [-1,1]-> [-1,1]
        Cpnts must be in the interval [-1,1].
        Parameters
        ----------
        Cpnts : ndarray
            grid points to transform.
        beta : float
            parameter to control the grid spacing.
        Returns
        -------
        Cpnts_mapped : ndarray
            transformed grid points
        dgdy: derivative of the coordinate transformation with respect to y (Cpnts)
        '''
        beta = float(beta)
        if beta == 0.0:
            Cpnts_mapped = Pnts
            dgdy = np.full(Pnts.shape,1.0)
        else:
            Cpnts_mapped = np.arcsin((beta)*Pnts)/np.arcsin(beta)
            dgdy = (beta/np.arcsin(beta))* 1.0/np.sqrt(1 - (beta*Pnts)**2 )
        return Cpnts_mapped, dgdy
