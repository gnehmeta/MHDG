#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import jacobiPol as jp
#****************************************************************************************
    


#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************    
def orthopoly2D_rst(x = None,n = None): 

    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************
    # Computes the ortogonal base of 2D polynomials of degree less
    # or equal to n at the point x=(r,s) in [-1,1]^2
    #************************************************************************************
    
    N = int((n + 1) * (n + 2) / 2)
    p = np.matrix(np.zeros((N,x.shape[0])))
    r = np.matrix(x[:,0])
    s = np.matrix(x[:,1])
    ncount = -1
    
    for nDeg in range(n+1):
        for i in range(nDeg+1):
            if i == 0:
                pi = np.matrix(np.ones((x.shape[0],1)))
                qi = np.matrix(np.ones((x.shape[0],1)))
            else:
                pi = np.matrix(jp.jacobiP(r,0,0,i))
                if pi.shape[1] != 1:
                    pi = np.transpose(pi) 
                qi = np.multiply(qi,(1 - s)) / 2
            j = nDeg - i
            if j == 0:
                pj = 1
            else:
                pj = jp.jacobiP(s,2 * i + 1,0,j)
            ncount = ncount + 1
            factor = np.sqrt((2 * i + 1) * (i + j + 1) / 2)
            mp = (np.multiply(np.multiply(pi,qi),pj)) * factor
            p[ncount,:] = np.transpose(mp[:,0])
    return p
#****************************************************************************************     