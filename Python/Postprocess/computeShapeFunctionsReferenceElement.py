#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import gaussLegendre as gl
import Vandermonde_LP as vm
import orthopoly1D_deriv as oly1d
import orthopoly2D_deriv_xieta as oly2dder
import GaussLegendreCubature2D as glc2
from scipy.linalg import lu
#****************************************************************************************


#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************    
def computeShapeFunctionsReferenceElement(nDeg = None,coord = None,nOfGaussPoints = None,
                                        varargin = ""):
    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************                                         
    # Function to compute the shape functions (& derivatives) at Gauss points   
    # Input:
    # nDeg:  degree of interpolation
    # coord: nodal coordinates at the reference element
    # nOfGaussPoints: n� of gauss points of the 1D quadrature
    # elementType (optional): 0 for quadrilateral, 1 for triangle. If it isn't
    #                         given only triangle or 1D elements are considered    
    # Output:
    # shapeFunctions: shape functions evaluated at the gauss points
    #                 size is nOfNodes X nOfGauss X (nsd + 1)
    #                 nsd+1 because shape function (1)
    #                 and derivatives (nsd) are stored
    # gaussWeights:   weights
    #************************************************************************************ 
    
    #************************************************************************************
    # Definition 1                                                                      *
    #************************************************************************************ 
    def computeShapeFunctionsReferenceElement1D(nDeg = None,coord = None,
                                                nOfGaussPoints = None): 
        #number of nodes/polynomials
        nOfNodes = nDeg + 1
        if nOfNodes != coord.shape[0]:
            raise Exception('Error computeShapeFunctionsReferenceElement1D')
        z,w = gl.gaussLegendre(nOfGaussPoints,- 1,1)
        nOfGauss = w.shape[1]
        #Vandermonde matrix
        V = vm.Vandermonde_LP(nDeg,coord)
        P,L,U = lu(np.transpose(V))
        P = np.linalg.inv(P)
        shapeFunctions = np.zeros((2,nOfNodes,nOfGauss))
        gaussWeights = np.transpose(w)
        gaussPoints = np.zeros((nOfGauss,1))
        #Integration over [-1,1]
        for i in range(nOfGauss):
            x = z[0,i]
            p,p_xi = oly1d.orthopoly1D_deriv(x,nDeg)
            mat = np.matrix(np.zeros((p.shape[0],2)))
            for j in range(p.shape[0]):
                mat[j,0]=p[j,0]
                mat[j,1]=p_xi[j,0]    
            N = np.linalg.solve(U,(np.linalg.solve(L,(np.dot(P,mat)))))
            N = np.matrix(np.transpose(N))
            shapeFunctions[0,:,i] = N[0,:]
            shapeFunctions[1,:,i] = N[1,:]
            # only for PFEM
            gaussPoints[i] = x       
        return shapeFunctions,gaussWeights,gaussPoints
    #************************************************************************************

    #************************************************************************************
    # Definition 2                                                                      *
    #************************************************************************************     
    def computeShapeFunctionsReferenceElement2D(nDeg = None,coord = None,
                                                nOfGaussPoints = None): 
        #number of nodes/polynomials
        nOfNodes = (nDeg + 1) * (nDeg + 2) / 2
        if nOfNodes != coord.shape[1-1]:
            raise Exception('Error computeShapeFunctionsReferenceElement2D')
        
        if nDeg < 12 and nOfGaussPoints - 2 < 11:
            if nOfGaussPoints==nDeg + 2:
                # NEW CALL FOR CUBATURES
                if 1 == nDeg:
                    OrderCubature = 5
                else:
                    if 2 == nDeg:
                        OrderCubature = 10
                    else:
                        if 3 == nDeg:
                            OrderCubature = 10
                        else:
                            if 4 == nDeg:
                                OrderCubature = 15
                            else:
                                if 5 == nDeg:
                                    OrderCubature = 15
                                else:
                                    if 6 == nDeg:
                                        OrderCubature = 15
                                    else:
                                        if 7 == nDeg:
                                            OrderCubature = 15
                                        else:
                                            if np.array([8,9,10,11]) == nDeg:
                                                OrderCubature = 25
            else:
                if 1 == nOfGaussPoints - 2:
                    OrderCubature = 5
                else:
                    if 2 == nOfGaussPoints - 2:
                        OrderCubature = 10
                    else:
                        if 3 == nOfGaussPoints - 2:
                            OrderCubature = 10
                        else:
                            if 4 == nOfGaussPoints - 2:
                                OrderCubature = 15
                            else:
                                if 5 == nOfGaussPoints - 2:
                                    OrderCubature = 15
                                else:
                                    if 6 == nOfGaussPoints - 2:
                                        OrderCubature = 15
                                    else:
                                        if 7 == nOfGaussPoints - 2:
                                            OrderCubature = 15
                                        else:
                                            if np.array([8,9,10,11]) == nOfGaussPoints - 2:
                                                OrderCubature = 25
            z,w = glc2.GaussLegendreCubature2D(OrderCubature)
            w = 2 * w
            z = 2 * z - 1
            nIP = len(w)
            nOfGauss = nIP
            #Vandermonde matrix
            V = vm.Vandermonde_LP(nDeg,coord)
            P,L,U = lu(np.transpose(V))
            P = np.linalg.inv(P)
            shapeFunctions = np.zeros((3,int(nOfNodes),int(nOfGauss)))
            gaussWeights = np.zeros((nOfGauss,1))
            gaussPoints = np.zeros((nOfGauss,2))
            #Integration over [-1,1]^2 using the cubature
            for i in range(nIP):
                x = z[i,:]
                p,p_xi,p_eta = oly2dder.orthopoly2D_deriv_xieta(x,nDeg)
                mat = np.matrix(np.zeros((p.shape[0],3)))
                for j in range(p.shape[0]):
                    mat[j,0]=p[j,0]
                    mat[j,1]=p_xi[j,0]
                    mat[j,2]=p_eta[j,0]
                N = np.linalg.solve(U,(np.linalg.solve(L,(np.dot(P,mat)))))
                N = np.matrix(np.transpose(N))
                shapeFunctions[0,:,i] = N[0,:]
                shapeFunctions[1,:,i] = N[1,:]
                shapeFunctions[2,:,i] = N[2,:]
                gaussWeights[i] = w[i]  
                gaussPoints[i]  = x
        else:
            z,w = gaussLegendre(nOfGaussPoints,- 1,1)
            nIP = len(w)
            nOfGauss = nIP ** 2
            #Vandermonde matrix
            V = Vandermonde_LP(nDeg,coord)
            L,U,P = lu(np.transpose(V))
            shapeFunctions = np.zeros((nOfNodes,nOfGauss,3))
            gaussWeights = np.zeros((nOfGauss,1))
            gaussPoints = np.zeros((nOfGauss,2))
            iGauss = 1
            #Integration over [-1,1]^2
            for i in np.arange(1,nIP+1).reshape(-1):
                for j in np.arange(1,nIP+1).reshape(-1):
                    x = np.array([z(i),z(j)])
                    p,p_xi,p_eta = orthopoly2D_deriv_rst(x,nDeg)
                    N = np.linalg.solve(U,(np.linalg.solve(L,(P * np.array([p,p_xi,p_eta])))))
                    shapeFunctions[:,iGauss,1] = np.transpose(N[:,1])
                    shapeFunctions[:,iGauss,2] = np.transpose(N[:,2])
                    shapeFunctions[:,iGauss,3] = np.transpose(N[:,3])
                    gaussWeights[iGauss] = (w(i) * w(j)) * (1 - x(2)) / 2
                    # only for PFEM
                    r = x(1)
                    s = x(2)
                    xi = (1 + r) * (1 - s) / 2 - 1
                    gaussPoints[iGauss,:] = np.array([xi,s])
                    iGauss = iGauss + 1
        return shapeFunctions,gaussWeights,gaussPoints        
    #************************************************************************************    
    
    #************************************************************************************
    # Definition 3 (Incomplete)                                                         *
    #************************************************************************************ 
    def computeShapeFunctionsReferenceElement3D(nDeg = None,coord = None,
                                                nOfGaussPoints = None): 
        #number of nodes/polynomials
        nOfNodes = (nDeg + 1) * (nDeg + 2) * (nDeg + 3) / 6
        if nOfNodes != coord.shape[1-1]:
            raise Exception('Error computeShapeFunctionsReferenceElement3D')
        z,w = gaussLegendre(nOfGaussPoints,- 1,1)
        nIP = len(w)
        nOfGauss = nIP ** 3
        #Vandermonde matrix
        V = Vandermonde_LP(nDeg,coord)
        L,U,P = lu(np.transpose(V))
        shapeFunctions = np.zeros((nOfNodes,nOfGauss,4))
        gaussWeights = np.zeros((nOfGauss,1))
        gaussPoints = np.zeros((nOfGauss,3))
        iGauss = 1
        #Integration over [-1,1]^3
        for i in np.arange(1,nIP+1).reshape(-1):
            for j in np.arange(1,nIP+1).reshape(-1):
                for k in np.arange(1,nIP+1).reshape(-1):
                    x = np.array([z(i),z(j),z(k)])
                    p,p_xi,p_eta,p_zeta = orthopoly3D_deriv_rst(x,nDeg)
                    N = np.linalg.solve(U,(np.linalg.solve(L,(P * np.array([p,p_xi,p_eta,p_zeta])))))
                    shapeFunctions[:,iGauss,1] = np.transpose(N[:,1])
                    shapeFunctions[:,iGauss,2] = np.transpose(N[:,2])
                    shapeFunctions[:,iGauss,3] = np.transpose(N[:,3])
                    shapeFunctions[:,iGauss,4] = np.transpose(N[:,4])
                    gaussWeights[iGauss] = (w(i)*w(j)*w(k))*((1-x(2))/2)*((1-x(3))/2)**2
                    # only for PFEM
                    r = x(1)
                    s = x(2)
                    t = x(3)
                    eta = (1 / 2) * (s - s * t - 1 - t)
                    xi = - (1 / 2) * (r + 1) * (eta + t) - 1
                    gaussPoints[iGauss,:] = np.array([xi,eta,t])
                    iGauss = iGauss + 1
        return shapeFunctions,gaussWeights,gaussPoints
    #************************************************************************************     
    if not len(str(varargin)) == 0 :
        elementType = str(varargin)[:]
        if elementType == 0:
            shapeFunctions,gaussWeights,gaussPoints = computeShapeFunctionsQua(nDeg,
            coord,nOfGaussPoints)
            return shapeFunctions,gaussWeights,gaussPoints                            
    if len(coord.shape) == 1:
        coord = coord.reshape(len(coord),1)
    nsd = coord.shape[1]
    if nsd == 1:
        shapeFunctions,gaussWeights,gaussPoints = computeShapeFunctionsReferenceElement1D(nDeg,coord,nOfGaussPoints)
    else:
        if nsd == 2:
            shapeFunctions,gaussWeights,gaussPoints = computeShapeFunctionsReferenceElement2D(nDeg,coord,nOfGaussPoints)
        else:
            if nsd == 3:
                shapeFunctions,gaussWeights,gaussPoints = computeShapeFunctionsReferenceElement3D(nDeg,coord,nOfGaussPoints)
            else:
                raise Exception('wrong nsd in computeShapeFunctionsReferenceElement')  
    return shapeFunctions,gaussWeights,gaussPoints  
#****************************************************************************************              
                  
