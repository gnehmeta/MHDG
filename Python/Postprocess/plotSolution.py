#****************************************************************************************************
# Modules                                                                                           *
#****************************************************************************************************
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import Vandermonde_LP as v
import orthopoly1D as o1d
import orthopoly2D as o2d
import getPermutationsQuads as gpq
import matplotlib.tri as tria
import scipy.spatial.qhull as qhull
from createShapeFunctions2dTensor import colsToCol
from createShapeFunctions2dTensor import createShapeFunctions2dTensor
from scipy.spatial import Delaunay
from numpy.linalg import inv
from math import log10
from pyhull.delaunay import DelaunayTri
#****************************************************************************************************

        

#****************************************************************************************************
# Definition                                                                                       *
#**************************************************************************************************** 
def plotSolution(X = None, lscale = None, T = None,u = None,refEl = None,nDegRef = None,cont = None,
                 logar = None,plotSuper = None,cbaxis = None, Max = None, Min = None): 

    # Plot Triangulation reference element 
    plotTriRefElem = 0

    u = np.matrix(u)
    if u.shape[1] > 1:
        raise Exception('Only 1d solutions')
    X = X[:,:]/lscale[0]      
    
    # Check input
    if nDegRef == None and cont == None and logar == None and plotSuper == None:
        nDegRef = 20
        cont = 0
        logar = 0
        plotSuper = 0
    else:
        if cont == None and logar == None and plotSuper == None:
            cont = 0
            logar = 0
            plotSuper = 0
        else:
            if logar == None and plotSuper == None:
                logar = 0
                plotSuper = 0
            else:
                if plotSuper == None:
                    plotSuper = 0
    
    # Plotting element (equal spaced points) 
    h = 1 / nDegRef
    if refEl['elemType'] == 1:
        nodes = []
        for j in range(nDegRef+1):
            i = np.transpose(np.matrix(np.arange(0,nDegRef - j+1)))
            aux = j * np.ones((i.shape[0],1))
            for k in range(i.shape[0]):
                nodes.append([i[k,0]*h,aux[k,0]*h])
        nodes = np.matrix(nodes) 
        nodes = np.multiply(2,nodes) - 1       
    else:
        if refEl['elemType'] == 0:
            coord1d = np.arange(- 1,1+2 * h,2 * h)
            nodesx,nodesy = np.meshgrid(coord1d,coord1d)
            nodes = np.zeros((nodesx.shape[0]*nodesx.shape[1], 2))
            nodes1 = colsToCol(nodesx)
            nodes2 = colsToCol(nodesy)
            for i in range(len(nodes)):
                nodes[i,0] = 2*nodes1[i,0]-1
                nodes[i,1] = 2*nodes2[i,0]-1 

    npoints = len(nodes)
    # Delaunay triangulation of plotting element
    if (refEl['elemType'] == 0 and nDegRef == 1):
        elemTriRef = np.array([[1,3,4],[1,4,2]])
    else:
        nodes11 = []
        nodes22 = [] 
        for i in range(nodes.shape[0]):
            nodes11.append(nodes[i,0]) 
            nodes22.append(nodes[i,1])

        elemTriRef = Delaunay(nodes)
        
        if plotTriRefElem:
            plt.figure()
            triangulation1 = tria.Triangulation(nodes11, nodes22, elemTriRef.simplices)
            plt.triplot(triangulation1, '-k')
            for i in range(len(nodes11)):
                plt.text(nodes11[i] , nodes22[i], i+1, fontSize = 10, color = 'r')
            plt.show()

    if refEl['elemType'] == 1:
        # Vandermonde matrix
        coordRef = np.matrix(refEl['NodesCoord'])
        nOfNodes = coordRef.shape[0]
        nDeg = refEl['degree'] 
        V = v.Vandermonde_LP(nDeg,coordRef)
        invV = inv(np.transpose(V))
        # Compute shape functions at interpolation points
        shapeFunctions = np.zeros((npoints,nOfNodes))
        for ipoint in range(npoints):
            p = o2d.orthopoly2D(nodes[ipoint,:],nDeg)
            shapeFunctions[ipoint,:] = np.transpose((invV@p))
    else:
        if refEl['elemType'] == 0:
            nOfNodes = refEl['NodesCoord'].shape[0]
            #Vandermonde matrix
            coordRef = np.transpose(np.matrix(refEl['NodesCoord1d']))
            nOfNodes1d = coordRef.shape[0]
            nDeg = refEl['degree'] 
            V = v.Vandermonde_LP(nDeg,coordRef)
            invV = inv(np.transpose(V))
            # Compute shape functions at interpolation points
            sf1d = np.zeros((np.asarray(coord1d).size,nOfNodes1d))
            for ipoint in range(len(coord1d)):
                p = o1d.orthopoly1D(coord1d[ipoint],nDeg)
                sf1d[ipoint,:] = np.transpose((invV@p))
            # permutations
            perm = gpq.getPermutationsQuads(nDeg)
            catSf1dT = np.zeros((2,np.transpose(sf1d).shape[0] ,np.transpose(sf1d).shape[1]))
            catSf1dT[0,:,:] = np.transpose(sf1d)
            catSf1dT[1,:,:] = np.transpose(sf1d) 
            shapeFunctions,gw2d,gp2d = createShapeFunctions2dTensor(catSf1dT,
                                     np.zeros((1,len(coord1d))),coord1d,perm)
            shapeFunctions = np.matrix(np.transpose(shapeFunctions[0,:,:]))
    
    #Delaunay's solution mesh
    nOfElemTriRef = len(elemTriRef.simplices)
    #nOfElemTriRef = elemTriRef.shape[0]
    nEl = T.shape[0]
    tri = np.matrix(np.zeros((nOfElemTriRef * nEl,3)))
    indexElem = 0
    uplot = np.zeros((nEl * npoints,1))
    Xplot = np.zeros((nEl * npoints,2))
    Xplot1 = [0]*nEl*npoints
    Xplot2 = [0]*nEl*npoints  
    uplot  = [0]*nEl*npoints 
    X = np.transpose(X)
    for ielem in range(nEl):
        Te = T[ielem,:]
        
        if np.asarray(u).size == X.shape[0]:
            # Continuous solution
            ueplot = shapeFunctions * u(Te)
        else:
            if np.asarray(u).size == np.asarray(T).size:
                # Discontinuous solution
                ind = np.arange(((ielem) * nOfNodes + 1),((ielem + 1) * nOfNodes)+1)
                u_ind = np.matrix(np.zeros((len(ind),1)))
                for i in range(len(ind)):
                    u_ind[i,0] = u[ind[i]-1,0]  
                ueplot = np.matmul(shapeFunctions,u_ind[:,0]) 
            else:
                raise Exception('Dimension of u is wrong')

        X_Te = np.matrix(np.zeros((Te.shape[0],X.shape[1])))

        for i in range(Te.shape[0]):
            X_Te[i,:] = X[int(Te[i]-1),:]  

        Xeplot = np.matmul(shapeFunctions,X_Te)   
        for ielemRef in range(nOfElemTriRef):
            indexElemRef = indexElem + ielemRef
            tri[indexElemRef,:] = (elemTriRef.simplices[ielemRef,:] + 
                                ielem * npoints * np.ones((1,elemTriRef.simplices.shape[1])))
            for i in range(tri.shape[1]):
                Xplot1[int(tri[indexElemRef,i])] = Xeplot[int(elemTriRef.simplices[ielemRef,i]),0]
                Xplot2[int(tri[indexElemRef,i])] = Xeplot[int(elemTriRef.simplices[ielemRef,i]),1]
                uplot [int(tri[indexElemRef,i])] = ueplot[int(elemTriRef.simplices[ielemRef,i]),0] 
        indexElem = indexElem + nOfElemTriRef     
    
    if cont == 1:
        '''
        Prova.Elements = tri
        Prova.Coordinates = Xplot
        qq,qq1 = tricontour_h(Prova,uplot,20)
        hold('on')
        Tb = Mesh.Tb
        for j in np.arange(1,Tb.shape[1-1]+1).reshape(-1):
            Tf = Tb[j,:]
            Xf = X[Tf,:] 
            plt.plot(Xf[:,1],Xf[:,2],'k-','LineWidth',1.5)
        plt.axis('equal')
        colormap('jet')
        colorbar('location','East')
        '''
    else:
        #Plot
        if logar:
            #uplot[uplot < 0] = 1e-15
            triangulation = tria.Triangulation(Xplot1, Xplot2, tri)
            plt.style.use('classic')
            Min = np.log10(Min)
            Max = np.log10(Max)
            plt.tricontourf(triangulation, np.log10(uplot), cmap ='plasma',
            levels = np.arange(Min,Max+(Max-Min)/20,(Max-Min)/20), vmax = Max, vmin = Min)
        else:
            triangulation = tria.Triangulation(Xplot1, Xplot2, tri)
            plt.style.use('classic')    
            plt.tricontourf(triangulation, uplot, cmap ='plasma',
            levels = np.arange(Min,Max+(Max-Min)/20,(Max-Min)/20) , vmax = Max, vmin = Min)

        if cont == 2:
            caxis(np.array([cbaxis]))              
#****************************************************************************************************

    