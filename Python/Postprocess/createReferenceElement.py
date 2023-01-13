#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import computeShapeFunctionsReferenceElement as csfre
import createShapeFunctions2dTensor as csf2T
import feketeNodes1D as fn1
import scipy.io
#****************************************************************************************



#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************   
def createReferenceElement(elementType, nOfElementNodes, handle = None, varargin = None): 
    #************************************************************************************
    # Description                                                                       *
    #************************************************************************************
    # theReferenceElement=createReferenceElement(elementType,nOfElementNodes)
    # Input:
    #  elementType: 0 for quadrilateral, 1 for triangle
    #  nOfElementNodes: number of nodes in the reference element
    #  nOfGaussPoints (optional): n of gauss points of the 1D quadrature. The
    #                             default value is nDeg + 2 where nDeg is the
    #                             degree of interpolation
    # Output:
    #  theReferenceElement: struct containing
    #     .IPcoordinates: coordinates of the integration points for 2D elemens
    #     .IPweights: weights of the integration points for 2D elements
    #     .N: shape functions at the IP
    #     .Nxi,.Neta: derivatives of the shape functions at the IP
    #     .IPcoordinates1d: coordinates of the integration points for 1D boundary elemens
    #     .IPweights1d: weights of the integration points for 1D boundary elements
    #     .N1d: 1D shape functions at the IP
    #     .N1dxi: derivatives of the 1D shape functions at the IP
    #     .faceNodes: matrix [nOfFaces nOfNodesPerFace] with the edge nodes numbering
    #     .innerNodes: vector [1 nOfInnerNodes] with the inner nodes numbering
    #     .faceNodes1d: vector [1 nOfNodesPerElement] with the 1D nodes numbering
    #     .NodesCoord: spatial coordinates of the element nodes
    #     .NodesCoord1d: spatial coordinates of the 1D element nodes
    #************************************************************************************
    
    if elementType == 1:
        if 3 == nOfElementNodes:
            nDeg = 1
            faceNodes = np.array([[1,2],[2,3],[3,1]])
            innerNodes = []
            faceNodes1d = np.arange(1,3)
            coord2d = np.array([[- 1,- 1],[1,- 1],[- 1,1]])
            coord1d = np.array([[- 1],[1]])
        else:
            if 6 == nOfElementNodes:
                nDeg = 2
                faceNodes = np.array([[1,4,2],[2,5,3],[3,6,1]])
                innerNodes = []
                faceNodes1d = np.arange(1,4)
                coord2d = np.array([[- 1,- 1],[1,- 1],[- 1,1],[0,- 1],[0,0],[- 1,0]])
                coord1d = np.array([[- 1],[0],[1]])
            else:
                if 10 == nOfElementNodes:
                    nDeg = 3
                    faceNodes = np.array([[1,4,5,2],[2,6,7,3],[3,8,9,1]])
                    innerNodes = 10
                    faceNodes1d = np.arange(1,5)
                else:
                    if 15 == nOfElementNodes:
                        nDeg = 4
                        faceNodes = np.array([[1,4,5,6,2],[2,7,8,9,3],[3,10,11,12,1]])
                        innerNodes = np.arange(13,16)
                        faceNodes1d = np.arange(1,6)
                    else:
                        if 21 == nOfElementNodes:
                            nDeg = 5
                            faceNodes = np.array([[1,np.arange(4,8),2],[2,np.arange(8,12),3],[3,np.arange(12,16),1]])
                            innerNodes = np.arange(16,22)
                            faceNodes1d = np.arange(1,7)
                        else:
                            if 28 == nOfElementNodes:
                                nDeg = 6
                                faceNodes = np.array([[1,np.arange(4,9),2],[2,np.arange(9,14),3],[3,np.arange(14,19),1]])
                                innerNodes = np.arange(19,29)
                                faceNodes1d = np.arange(1,8)
                            else:
                                if 36 == nOfElementNodes:
                                    nDeg = 7
                                    faceNodes = np.array([[1,np.arange(4,10),2],[2,np.arange(10,16),3],[3,np.arange(16,22),1]])
                                    innerNodes = np.arange(22,37)
                                    faceNodes1d = np.arange(1,9)
                                else:
                                    if 45 == nOfElementNodes:
                                        nDeg = 8
                                        faceNodes = np.array([[1,np.arange(4,11),2],[2,np.arange(11,18),3],[3,np.arange(18,25),1]])
                                        innerNodes = np.arange(25,46)
                                        faceNodes1d = np.arange(1,10)
                                    else:
                                        if 55 == nOfElementNodes:
                                            nDeg = 9
                                            faceNodes = np.array([[1,np.arange(4,12),2],[2,np.arange(12,20),3],[3,np.arange(20,28),1]])
                                            innerNodes = np.arange(28,56)
                                            faceNodes1d = np.arange(1,11)
                                        else:
                                            if 66 == nOfElementNodes:
                                                nDeg = 10
                                                faceNodes = np.array([[1,np.arange(4,13),2],[2,np.arange(13,22),3],[3,np.arange(22,31),1]])
                                                innerNodes = np.arange(31,67)
                                                faceNodes1d = np.arange(1,12)
                                            else:
                                                if 78 == nOfElementNodes:
                                                    nDeg = 11
                                                    faceNodes = np.array([[1,np.arange(4,14),2],[2,np.arange(14,24),3],[3,np.arange(24,34),1]])
                                                    innerNodes = np.arange(34,79)
                                                    faceNodes1d = np.arange(1,13)
                                                else:
                                                    raise Exception('Not implemented yet')
        if nDeg >= 3:
            feketeFile = scipy.io.loadmat('positionFeketeNodesTri2D_EZ4U.mat')
            coord2d = feketeFile['feketeNodesPosition']["".join(['P',str(nDeg)])]
            coord2d = coord2d[0,0]
            coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
    else:
        if elementType == 0:
            if 4 == nOfElementNodes:
                nDeg = 1
                faceNodes = np.array([[1,2],[2,3],[3,4],[4,1]])
                innerNodes = []
                faceNodes1d = np.arange(1,2+1)
                coord2d = np.array([[- 1,- 1],[1,- 1],[1,1],[- 1,1]])
                coord1d = np.array([[- 1],[1]])
                perm = np.array([1,2,4,3])
            else:
                if 9 == nOfElementNodes:
                    nDeg = 2
                    faceNodes = np.array([[1,5,2],[2,6,3],[3,7,4],[4,8,1]])
                    innerNodes = 9
                    faceNodes1d = np.arange(1,3+1)
                    coord2d = np.array([[- 1,- 1],[1,- 1],[1,1],[- 1,1],[0,- 1],[1,0],[0,1],[- 1,0],[0,0]])
                    coord1d = np.array([[- 1],[0],[1]])
                    perm = np.array([1,5,2,8,9,6,4,7,3])
                else:
                    if 16 == nOfElementNodes:
                        nDeg = 3
                        faceNodes = np.array([[1,5,6,2],[2,7,8,3],[3,9,10,4],[4,11,12,1]])
                        innerNodes = np.arange(13,17)
                        faceNodes1d = np.arange(1,5)
                        coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                        coord2d = np.zeros(((nDeg + 1) ** 2,2))
                        for i in range(1,nDeg + 2):
                            ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                            coord2d[ind-1,0] = coord1d
                            coord2d[ind-1,1] = coord1d[i-1] 
                        perm = np.array([1,5,6,2,12,13,14,7,11,15,16,8,4,10,9,3])
                        coord2d[perm-1,:] = coord2d
                    else:
                        if 25 == nOfElementNodes:
                            nDeg = 4
                            faceNodes = np.array([[1,5,6,7,2],[2,8,9,10,3],[3,11,12,13,4],[4,14,15,16,1]])
                            innerNodes = np.arange(17,26)
                            faceNodes1d = np.arange(1,6)
                            coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                            coord2d = np.zeros(((nDeg + 1) ** 2,2))
                            for i in range(1,nDeg + 2):
                                ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                                coord2d[ind-1,0] = coord1d
                                coord2d[ind-1,1] = coord1d[i-1]
                            perm = np.array([1,5,6,7,2,16,17,18,19,8,15,20,21,22,9,14,23,24,25,10,4,13,12,11,3])
                            coord2d[perm-1,:] = coord2d
                        else:
                            if 36 == nOfElementNodes:
                                nDeg = 5
                                faceNodes = np.array([[1,5,6,7,8,2],[2,9,10,11,12,3],[3,13,14,15,16,4],[4,17,18,19,20,1]])
                                innerNodes = np.arange(21,37)
                                faceNodes1d = np.arange(1,7)
                                coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                                coord2d = np.zeros(((nDeg + 1) ** 2,2))
                                for i in range(1,nDeg + 2):
                                    ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                                    coord2d[ind-1,0] = coord1d
                                    coord2d[ind-1,1] = coord1d[i-1] 
                                perm = np.array([1,5,6,7,8,2,20,21,22,23,24,9,19,25,26,27,28,10,18,29,30,31,32,11,17,33,34,35,36,12,4,16,15,14,13,3])
                                coord2d[perm-1,:] = coord2d
                            else:
                                if 49 == nOfElementNodes:
                                    nDeg = 6
                                    faceNodes = np.array([[1,5,6,7,8,9,2],[2,10,11,12,13,14,3],[3,15,16,17,18,19,4],[4,20,21,22,23,24,1]])
                                    innerNodes = np.arange(25,50)
                                    faceNodes1d = np.arange(1,8)
                                    coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                                    coord2d = np.zeros(((nDeg + 1) ** 2,2))
                                    for i in range(1,nDeg + 2):
                                        ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                                        coord2d[ind-1,0] = coord1d
                                        coord2d[ind-1,1] = coord1d[i-1] 
                                    perm = np.array([1,5,6,7,8,9,2,24,25,26,27,28,29,10,23,30,31,32,33,34,11,22,35,36,37,38,39,12,21,40,41,42,43,44,13,20,45,46,47,48,49,14,4,19,18,17,16,15,3])
                                    coord2d[perm-1,:] = coord2d
                                else:
                                    if 64 == nOfElementNodes:
                                        nDeg = 7
                                        faceNodes = np.array([[1,5,6,7,8,9,10,2],[2,11,12,13,14,15,16,3],[3,17,18,19,20,21,22,4],[4,23,24,25,26,27,28,1]])
                                        innerNodes = np.arange(29,65)
                                        faceNodes1d = np.arange(1,9)
                                        coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                                        coord2d = np.zeros(((nDeg + 1) ** 2,2))
                                        for i in range(1,nDeg + 2):
                                            ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                                            coord2d[ind-1,0] = coord1d
                                            coord2d[ind-1,1] = coord1d[i-1]
                                        perm = np.array([1,5,6,7,8,9,10,2,28,29,30,31,32,33,34,11,27,35,36,37,38,39,40,12,26,41,42,43,44,45,46,13,25,47,48,49,50,51,52,14,24,53,54,55,56,57,58,15,23,59,60,61,62,63,64,16,4,22,21,20,19,18,17,3])
                                        coord2d[perm-1,:] = coord2d
                                    else:
                                        if 81 == nOfElementNodes:
                                            nDeg = 8
                                            faceNodes = np.array([[1,5,6,7,8,9,10,11,2],[2,12,13,14,15,16,17,18,3],[3,19,20,21,22,23,24,25,4],[4,26,27,28,29,30,31,32,1]])
                                            innerNodes = np.arange(33,82)
                                            faceNodes1d = np.arange(1,10)
                                            coord1d = fn1.feketeNodes1D(nDeg,faceNodes1d)
                                            coord2d = np.zeros(((nDeg + 1) ** 2,2))
                                            for i in range(1,nDeg + 2):
                                                ind = (i - 1) * (nDeg + 1) + (np.arange(1,nDeg + 2))
                                                coord2d[ind-1,0] = coord1d
                                                coord2d[ind-1,1] = coord1d[i-1] 
                                            perm = np.array([1,5,6,7,8,9,10,11,2,32,33,34,35,36,37,38,39,12,31,40,41,42,43,44,45,46,13,30,47,48,49,50,51,52,53,14,29,54,55,56,57,58,59,60,15,28,61,62,63,64,65,66,67,16,27,68,69,70,71,72,73,74,17,26,75,76,77,78,79,80,81,18,4,25,24,23,22,21,20,19,3])
                                            coord2d[perm-1,:] = coord2d
                                        else:
                                            raise Exception('Not implemented yet')
        else:
            raise Exception('Element not allowed')
    
    #Compute shape functions and quadrature
    if varargin == None:
        nOfGaussPoints = nDeg + 2
        #     nOfGaussPoints = 2*nDeg + 1;
    else:
        nOfGaussPoints = varargin[:]
        
    
    if elementType == 1:
        shapeFun2d,gw2d,gp2d = csfre.computeShapeFunctionsReferenceElement(nDeg,coord2d,nOfGaussPoints,elementType)
        shapeFun1d,gw1d,gp1d = csfre.computeShapeFunctionsReferenceElement(nDeg,coord1d,nOfGaussPoints)
    else:
        if elementType == 0:
            shapeFun1d,gw1d,gp1d = csfre.computeShapeFunctionsReferenceElement(nDeg,coord1d,nOfGaussPoints)
            shapeFun2d,gw2d,gp2d = csf2T.createShapeFunctions2dTensor(shapeFun1d,gw1d,gp1d,perm)
            #[shapeFun2d,gw2d,gp2d] = ProvaComputeShapeFunctionsReferenceElementQuads(nDeg,coord1d,nOfGaussPoints);
    
    N = np.transpose(shapeFun2d[0,:,:])
    Nxi = np.transpose(shapeFun2d[1,:,:])
    Neta = np.transpose(shapeFun2d[2,:,:])
    N1 = np.transpose(shapeFun1d[0,:,:])
    Nxi1 = np.transpose(shapeFun1d[1,:,:])
    #Creating reference element structure
    theReferenceElement = dict([('IPcoordinates',gp2d),('IPweights',gw2d),('N',N)
    ,('Nxi',Nxi),('Neta',Neta)
    ,('IPcoordinates1d',gp1d),('IPweights1d',gw1d),('N1d',N1),('N1dxi',Nxi1)
    ,('faceNodes',faceNodes)
    ,('innerNodes',innerNodes),('faceNodes1d',faceNodes1d),('NodesCoord',coord2d)
    ,('NodesCoord1d',coord1d),('degree',nDeg),('elemType',elementType)])
    return theReferenceElement
#****************************************************************************************