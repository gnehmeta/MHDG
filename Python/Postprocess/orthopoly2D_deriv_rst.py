#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import jacobiPol as jp
#****************************************************************************************



#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************
def orthopoly2D_deriv_rst(x = None,n = None): 
    
    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************
    # Computes the ortogonal base of 2D polynomials of degree less
    # or equal to n at the point x=(r,s) in [-1,1]^2
    #************************************************************************************
    
    N       = int((n + 1)*(n + 2)/2)
    p       = np.matrix(np.zeros((N,x.shape[0])))
    dp_dxi  = np.matrix(np.zeros((N,x.shape[0])))
    dp_deta = np.matrix(np.zeros((N,x.shape[0])))
    r       = np.matrix(x[:,0])
    s       = np.matrix(x[:,1])
    xi      = np.multiply((1 + r),(1 - s))/2 - 1
    eta     = s
    dr_dxi  = np.divide(2.0,1-eta)
    dr_deta = np.divide(2*(1+xi),(1-eta)**2)
    ncount = -1
    
    for nDeg in range(n+1):
        for i in range(nDeg+1):
            if i == 0:
                p_i  = np.matrix(np.ones((x.shape[0],1)))
                q_i  = np.matrix(np.ones((x.shape[0],1)))
                dp_i = np.matrix(np.zeros((x.shape[0],1)))
                dq_i = np.matrix(np.zeros((x.shape[0],1)))
            else:
                p_i  = np.matrix(jp.jacobiP(r,0,0,i))
                dp_i = np.matrix(jp.jacobiP(r,1,1,i - 1))*(i+1)/2
                q_i  = np.multiply(q_i,(1 - s)) / 2
                dq_i = np.divide(np.multiply(q_i,(- i)), (1 - s))

            j = nDeg - i
            if j == 0:
                p_j  = np.matrix(np.ones((x.shape[0],1)))
                dp_j = np.matrix(np.zeros((x.shape[0],1)))
            else:
                p_j  = np.matrix(jp.jacobiP(s,2 * i + 1,0,j))
                dp_j = np.matrix(jp.jacobiP(s,2*i + 2,1,j-1))*(j+2*i+2)/2
            ncount = ncount + 1
            factor = np.sqrt((2 * i + 1) * (i + j + 1) / 2)
            p[ncount,:] = (np.multiply(np.multiply(p_i,q_i),p_j)) * factor
            dp_dr = (np.multiply(np.multiply((dp_i),q_i),p_j)) * factor
            dp_ds = (np.multiply(p_i,(np.multiply(dq_i,p_j) +
                     np.multiply(q_i,dp_j)))) * factor
            dp_dxi[ncount,:] = np.multiply(dp_dr,dr_dxi)
            dp_deta[ncount,:] = np.multiply(dp_dr,dr_deta) + dp_ds
    return p,dp_dxi,dp_deta
#****************************************************************************************    