#**************************************************************************************************
# Modules                                                                                         *
#**************************************************************************************************
import numpy as np
import orthopoly1D as oly1
import orthopoly2D as oly2
#**************************************************************************************************


#**************************************************************************************************
# Definition                                                                                      *
#**************************************************************************************************
def Vandermonde_LP(nDeg = None,coord = None): 

    #**********************************************************************************************
    # Description                                                                                 *
    #**********************************************************************************************    
    # Function to compute the Vandermonde matrix
        
    # Input:
    # nDeg : degree of polynomials
    # coord: nodal coordinates 
        
    # Output:
    # V: Vandemonde matrix
    #**********************************************************************************************
    

    #**********************************************************************************************
    # Definition (1D Matrix)                                                                      *
    #**********************************************************************************************   
    def Vandermonde_LP1D(nDeg = None,coord = None): 
        N = len(coord)
        
        if (N != nDeg + 1):
            raise Exception('The number of polynomials does not coincide with the number of nodes')
        
        V = np.zeros((N,N))
        for i in np.arange(1,N+1).reshape(-1):
            x = coord[i-1]
            p = oly1.orthopoly1D(x,nDeg)
            V[i-1,:] = np.transpose(p)
        return V  
    #**********************************************************************************************      
    
    
    #**********************************************************************************************
    # Definition (2D Matrix)                                                                      *
    #********************************************************************************************** 
    def Vandermonde_LP2D(nDeg = None,coord = None): 
        N = coord.shape[0]
        
        if (N != (nDeg + 1) * (nDeg + 2) / 2):
            raise Exception('The number of polynomials does not coincide with the number of nodes')
        
        x = coord
        p = oly2.orthopoly2D(x,nDeg)
        V = np.transpose(p)
        return V
    #**********************************************************************************************    
    

    #**********************************************************************************************
    # Definition (3D Matrix) INCOMPLETE!                                                          *
    #**********************************************************************************************    
    def Vandermonde_LP3D(nDeg = None,coord = None): 
        N = coord.shape[1-1]
        if (N != (nDeg + 1) * (nDeg + 2) * (nDeg + 3) / 6):
            raise Exception('The number of polynomials does not coincide with the number of nodes')
        
        V = np.zeros((N,N))
        for i in np.arange(1,N+1).reshape(-1):
            x = coord[i-1,:]
            p = orthopoly3D(x,nDeg)
            V[i-1,:] = np.transpose(p)
        return V
    #**********************************************************************************************     

    nsd = coord.shape[1]
    if nsd == 1:
        V = Vandermonde_LP1D(nDeg,coord)
    else:
        if nsd == 2:
            V = Vandermonde_LP2D(nDeg,coord)
        else:
            if nsd == 3:
                V = Vandermonde_LP3D(nDeg,coord)
            else:
                raise Exception('Vandermonde_LP requires coordinates in 1D, 2D or 3D')    
    
    return V  
#**************************************************************************************************      