#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import orthopoly2D_rst as oly2rst
#****************************************************************************************



#****************************************************************************************
# Definition 1                                                                          *
#****************************************************************************************
def find_indices(list_to_check, item_to_find):
    
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    
    return list(indices)
#****************************************************************************************    



#****************************************************************************************
# Definition 2                                                                          *
#****************************************************************************************
def orthopoly2D(x = None,n = None): 

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
    p = oly2rst.orthopoly2D_rst(mat,n)
    return p
#****************************************************************************************    