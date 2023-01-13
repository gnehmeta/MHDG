#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
#****************************************************************************************
    

#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************    
def gaussLegendre(N = None,a = None,b = None): 
    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************
    # This script is for computing definite integrals using Legendre-Gauss
    # Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
    # [a,b] with truncation order N    
    # Suppose you have a continuous function f(x) which is defined on [a,b]
    # which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    # the values contained in the x vector to obtain a vector f. Then compute
    # the definite integral using sum(f.*w)
    #************************************************************************************
    N  = N - 1
    N1 = N + 1
    N2 = N + 2
    xu = np.transpose(np.linspace(- 1,1,N1))
    # Initial guess
    y = np.transpose([np.cos((2 * np.transpose((np.arange(0,N+1))) + 1)*np.pi / (2 * N + 2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)])
    y = np.matrix(y)
    # Legendre-Gauss Vandermonde Matrix
    L = np.zeros((N1,N2))
    L = np.matrix(L)
    # Derivative of LGVM
    Lp = np.zeros((N1,N2))
    Lp = np.matrix(Lp)
    Lpp= np.matrix(np.zeros((N1,1)))
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0 = 2
    # Iterate until new points are uniformly within epsilon of old points
    while np.amax(np.abs(y - y0)) > np.finfo(float).eps:
       # Lp = Lp.reshape(1,-1)
        L[:,0] = np.ones((N1,1))
        Lp[:,0] = np.zeros((N1,1))
        L[:,1] = y[:,0]
        Lp[:,1] = np.ones((N1,1))
        for k in range(2,N1+1):
            L[:,k] = (np.multiply((2 * k - 1) * y,L[:,k-1]) - (k - 1) * L[:,k - 2]) / k    
        Lpp[:,0] = np.matrix((N2) * (L[:,N1-1] - np.multiply(y[:,0],L[:,N2-1])) / (np.ones((y.shape[0],1)) - np.power(y[:,0],2)))
        y0 = y
        y = y0 - L[:,N2-1] / Lpp[:,0]   
    # Linear map from[-1,1] to [a,b]
    x = (a * (1 - y) + b * (1 + y)) / 2
    # Compute the weights
    w = (b - a) / (np.multiply((1 - np.power(y,2)),np.power(Lpp,2))) * (N2 / N1) ** 2
    x = np.transpose(x)
    w = np.transpose(w)
    return x,w
#****************************************************************************************    
