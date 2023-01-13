#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import orthopoly2D_deriv_rst as oly2drst
from orthopoly2D import find_indices
#****************************************************************************************



#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************
def orthopoly2D_deriv_xieta(x = None,n = None): 

    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************
    # Computes the ortogonal base of 2D polynomials of degree less
    # or equal to n at the point x=(xi,eta) in the reference triangle
    #************************************************************************************
    if type(x) == np.matrix:
        x = np.array(x)
    xi  = x[:,0] 
    eta = x[:,1]
    r = np.zeros((x.shape[0],1)) 
    s = np.transpose(np.matrix(eta))
    arrange = [] 
    if list(eta).count(1)>0:
        arrange = find_indices(eta,1)
        r[arrange,0] = -1
        s[arrange,0] = 1
    for i in range(len(xi)):
        if arrange.count(i)==0:
            r[i,0]=2*(1+xi[i])/(1-eta[i])-1
        else:
            continue        
    mat = np.matrix(np.zeros((x.shape[0],2)))   
    mat[:,0] = r
    mat[:,1] = s       
    p,dp_dxi,dp_deta = oly2drst.orthopoly2D_deriv_rst(mat,n)
    return p,dp_dxi,dp_deta
#**************************************************************************************** 