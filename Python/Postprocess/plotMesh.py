#****************************************************************************************************
# Modules                                                                                           *
#****************************************************************************************************
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib.collections 
import matplotlib.tri as tri
#****************************************************************************************************


#****************************************************************************************************
# Definition 1                                                                                      *
#****************************************************************************************************    
def plotMesh(X = None,T = None,eltype = None, option = None,nodesNum = None): 
# Plots the mesh defined by X and T    
# Input:
#   X: nodal coordinates
#   T: connectivities (elements)
#   faceNodes: nOfFaces x nOfNodesPerFace matrix. Indicates the order of
#              the face nodes in a reference element. The numbering of
#              the faces has to be given in a clockwise sense, for instance:   
#              For a given face, the column index of the matrix indicates
#              the global position of the node. This global numbering has
#              to ascend in a clockwise sense too.    
#   option (optional): type 'plotNodes' to see the nodes' position on the
#                      ploted mesh, or type 'plotNodesNum' to see their global
#                      number.
#   nodesNum (necesary if option = 'plotNodesNum'): type 'all' to plot the
#                                                   global postion of all
#                                                   nodes, or enter a list
#                                                   array with the selected
#                                                   nodes.    
# Output:
#   patchHandle (optional): handle to the created patch object
    
    if eltype == 0:
        faceNodes = npy.matrix(faceNodes_aux_quads(T.shape[1]))
        optn = 1
    else:
        faceNodes = npy.matrix(faceNodes_aux(T.shape[1]))
        optn = 1
    
    #Ordering the face nodes in a row vector without connectivity between them
    nOfFaces,nOfNodesPerFace = faceNodes.shape
    np = nOfNodesPerFace - 1
    oFaceNodes = npy.matrix(npy.zeros((1,nOfFaces*np)))
    aux = 1 - np
    aux2 = 0
    for iface in range(nOfFaces):
        aux = aux + np
        aux2 = aux2 + np
        oFaceNodes[0,npy.arange(aux-1,aux2)] = faceNodes[iface,npy.arange(0,np)]
    
    #Conectivity for the faces
    T = npy.matrix(T)
    patchFaces = npy.matrix(npy.zeros((T.shape[0], oFaceNodes.shape[1])))
    for i in range(oFaceNodes.shape[1]):
        patchFaces[:,i] = T[:,int(oFaceNodes[0,i])-1]  

    #Plot mesh
    for i in range(patchFaces.shape[0]):
        xElement = []
        yElement = []
        for j in range(patchFaces.shape[1]):
            xElement.append(X[0,int(patchFaces[i,j])-1])
            yElement.append(X[1,int(patchFaces[i,j])-1])
        plt.plot(xElement,yElement, color = 'b')        
    
    """
    #Optional plots
    if len(varargin) > (2 + optn) and ischar(option):
        hold('on')
        if strcmpi(option,'plotNodes'):
            plt.plot(X[:,1],X[:,2],'o','markerSize',3,'markerFaceColor','b')
        else:
            if (len(varargin) == (4 + optn)) and strcmpi(option,'plotNodesNum'):
                if strcmpi(nodesNum,'all'):
                    list = npy.arange(1,X.shape[0]+1)
                    fontSize = 10
                else:
                    if not isnumeric(nodesNum) :
                        raise Exception('wrong list of nodes')
                    else:
                        list = nodesNum
                        fontSize = 15
                        plt.plot(X(list,1),X(list,2),'o','markerSize',3,'markerFaceColor','b')
                for inode in list.reshape(-1):
                    text(X(inode,1),X(inode,2),int2str(inode),'FontSize',fontSize,'Color',npy.array([1,0,0]))
            else:
                if (len(varargin) == (4 + optn)) and strcmpi(option,'plotNodesNumAndElements'):
                    if strcmpi(nodesNum,'all'):
                        list = npy.arange(1,X.shape[1-1]+1)
                        fontSize = 16
                    else:
                        if not isnumeric(nodesNum) :
                            raise Exception('wrong list of nodes')
                        else:
                            list = nodesNum
                            fontSize = 15
                            plt.plot(X(list,1),X(list,2),'o','markerSize',3,'markerFaceColor','b')
                    for inode in list.reshape(-1):
                        text(X(inode,1),X(inode,2),int2str(inode),'FontSize',fontSize,'Color',npy.array([1,0,0]))
                    for iElem in npy.arange(1,T.shape[1-1]+1).reshape(-1):
                        xbar = 1 / 3 * (X(T(iElem,1),1) + X(T(iElem,2),1) + X(T(iElem,3),1))
                        ybar = 1 / 3 * (X(T(iElem,1),2) + X(T(iElem,2),2) + X(T(iElem,3),2))
                        text(xbar,ybar,int2str(iElem),'FontSize',fontSize + 2,'Color',npy.array([0,0,1]))
                else:
                    if (len(varargin) == (3 + optn)) and strcmpi(option,'plotElements'):
                        fontSize = 15
                        for iElem in npy.arange(1,T.shape[1-1]+1).reshape(-1):
                            xbar = 1 / 3 * (X(T(iElem,1),1) + X(T(iElem,2),1) + X(T(iElem,3),1))
                            ybar = 1 / 3 * (X(T(iElem,1),2) + X(T(iElem,2),2) + X(T(iElem,3),2))
                            text(xbar,ybar,int2str(iElem),'FontSize',fontSize + 2,'Color',npy.array([0,0,1]))
                    else:
                        raise Exception('wrong optional argument. Check help to fix the error')
        hold = 'off' 
    
    #Output variable
    if not nargout :
        varargout = npy.array([])
    else:
        varargout = npy.array([patchHandle])

    return varargout 
    """  
#****************************************************************************************************     



#****************************************************************************************************
# Definition 2                                                                                      *
#****************************************************************************************************     
def faceNodes_aux(nOfElementNodes = None): 
    if 3 == nOfElementNodes:
        res = npy.array([[1,2],[2,3],[3,1]])
    else:
        if 6 == nOfElementNodes:
            res = npy.array([[1,4,2],[2,5,3],[3,6,1]])
        else:
            if 10 == nOfElementNodes:
                res = npy.array([[1,4,5,2],[2,6,7,3],[3,8,9,1]])
            else:
                if 15 == nOfElementNodes:
                    res = npy.array([[1,4,5,6,2],[2,7,8,9,3],[3,10,11,12,1]])
                else:
                    if 21 == nOfElementNodes:
                        res = npy.array([[1,npy.arange(4,7+1),2],[2,npy.arange(8,11+1),3],[3,npy.arange(12,15+1),1]])
                    else:
                        if 28 == nOfElementNodes:
                            res = npy.array([[1,npy.arange(4,8+1),2],[2,npy.arange(9,13+1),3],[3,npy.arange(14,18+1),1]])
                        else:
                            if 36 == nOfElementNodes:
                                res = npy.array([[1,npy.arange(4,9+1),2],[2,npy.arange(10,15+1),3],[3,npy.arange(16,21+1),1]])
                            else:
                                if 45 == nOfElementNodes:
                                    res = npy.array([[1,npy.arange(4,10+1),2],[2,npy.arange(11,17+1),3],[3,npy.arange(18,24+1),1]])
                                else:
                                    if 55 == nOfElementNodes:
                                        res = npy.array([[1,npy.arange(4,11+1),2],[2,npy.arange(12,19+1),3],[3,npy.arange(20,27+1),1]])
                                    else:
                                        if 66 == nOfElementNodes:
                                            res = npy.array([[1,npy.arange(4,12+1),2],[2,npy.arange(13,21+1),3],[3,npy.arange(22,30+1),1]])
                                        else:
                                            if 78 == nOfElementNodes:
                                                res = npy.array([[1,npy.arange(4,13+1),2],[2,npy.arange(14,23+1),3],[3,npy.arange(24,33+1),1]])
    
    return res  
#****************************************************************************************************                                              
    


#****************************************************************************************************
# Definition 3                                                                                      *
#****************************************************************************************************     
def faceNodes_aux_quads(nOfElementNodes = None): 
    if 4 == nOfElementNodes:
        res = npy.array([[1,2],[2,3],[3,4],[4,1]])
    else:
        if 9 == nOfElementNodes:
            res = npy.array([[1,5,2],[2,6,3],[3,7,4],[4,8,1]])
        else:
            if 16 == nOfElementNodes:
                res = npy.array([[1,5,6,2],[2,7,8,3],[3,9,10,4],[4,11,12,1]])
            else:
                if 25 == nOfElementNodes:
                    res = npy.array([[1,5,6,7,2],[2,8,9,10,3],[3,11,12,13,4],[4,14,15,16,1]])
                else:
                    if 36 == nOfElementNodes:
                        res = npy.array([[1,5,6,7,8,2],[2,9,10,11,12,3],[3,13,14,15,16,4],[4,17,18,19,20,1]])
                    else:
                        if 49 == nOfElementNodes:
                            res = npy.array([[1,5,6,7,8,9,2],[2,10,11,12,13,14,3],[3,15,16,17,18,19,4],[4,20,21,22,23,24,1]])
                        else:
                            if 64 == nOfElementNodes:
                                res = npy.array([[1,5,6,7,8,9,10,2],[2,11,12,13,14,15,16,3],[3,17,18,19,20,21,22,4],[4,23,24,25,26,27,28,1]])
                            else:
                                if 81 == nOfElementNodes:
                                    res = npy.array([[1,5,6,7,8,9,10,11,2],[2,12,13,14,15,16,17,18,3],[3,19,20,21,22,23,24,25,4],[4,26,27,28,29,30,31,32,1]])
    
    return res
#****************************************************************************************************    


