#*******************************************************************************************************************************
# Modules                                                                                                                      *
#*******************************************************************************************************************************
import HDF5load as h
import numpy as np
import math as m
import plotFortranSolution as pfs
import cons2phys as c2p
import setSubPlot as sp
import createReferenceElement as cre
import matplotlib.pyplot as plt
import plotMesh as pm
from plotSolution_PAdapt import plotSolution_PAdapt
from colorama import Fore
#*******************************************************************************************************************************




#********************************************************************************************************************************
# Initial message                                                                                                               *
#********************************************************************************************************************************   
print(Fore.CYAN + "\033[1m" + '#*********************************************************************************'+ "\033[0m")
print(Fore.CYAN + "\033[1m" + 'Python Version of plotFortranSolution \t\t\t\t\t\t *' + "\033[0m")
print(Fore.CYAN + "\033[1m" + '#*********************************************************************************' + "\033[0m")
rmkMsg = """REMARKS: 1. If you have a HDF5 error, please load the module: anaconda/python37  
    \t 2. Make sure you have the right process number (nproc),
    \t the right path for your Mesh (meshPath),
    \t the right path for your solution (solPath),
    \t and the right name for the solution (solName) in plotFortranSolution.py"""
print(Fore.RED + "\033[1m"+ rmkMsg +"\033[0m")
print(Fore.CYAN +"\033[1m"+'[GETTING READY...]'+"\033[0m")

#*******************************************************************************************************************************
# Load solutions                                                                                                               *
#*******************************************************************************************************************************
ndim = pfs.Data['ndim'][0]
nodes = pfs.Data['nodes'][0]
NrefElts = pfs.Data['NrefElts'][0]
Nelts_per_refElt = pfs.Data['Nelts_per_refElt']
eltType = pfs.Data['typ_of_elt']
elts_refElt_map = pfs.Data['elts_refElt_map']
refElts_deg = pfs.Data['refElts_deg']
nnodesAllEltsRefElts = pfs.Data['nnodesAllEltsRefElts']
U = pfs.Data['U']
nnodes_PAdapt_in_element = pfs.Data['nnodes_PAdapt_in_element']
nodes_PAdapt = pfs.Data['nodes_PAdapt'][:] 
nodes_PAdapt = np.transpose(np.matrix(nodes_PAdapt))
L = pfs.Data['L']
Dirichlet = pfs.Data['Dirichlet']
Q = pfs.Data['Q']
Velocity = pfs.Data['Velocity']
DiffusionTensor = pfs.Data['DiffusionTensor']

U2 = pfs.Data1['U']
L2 = pfs.Data1['L']
Dirichlet2 = pfs.Data1['Dirichlet']
Q2 = pfs.Data1['Q']
Velocity2 = pfs.Data1['Velocity']
DiffusionTensor2 = pfs.Data1['DiffusionTensor']
#*******************************************************************************************************************************



#*******************************************************************************************************************************
# Definition                                                                                                                   *
#*******************************************************************************************************************************
def boundariesColorBar(fig):
    Max = 0 
    Min = 10**100
    refEltType_irefElt = None
    nnodes_PAdapt_per_refElt_irefElt = None
    class Dim:
        X_padapt    = np.matrix(np.zeros((nodes_PAdapt.shape[0],1)))
        x_padapt    = [] 
        XElt_padapt = [] 
    
    for irefElt in range (1,NrefElts+1):
        X_mesh = nodes
        dims = [] 
        for idim in range(ndim):
            dim = Dim()
            dim.X_padapt = nodes_PAdapt[:,idim] 
            dims.append(dim)   
        u = []
        for idim in range(ndim):
            dims[idim].x_padapt = [] 
        count = 0
        for ielt in range(sum(Nelts_per_refElt)):
            if elts_refElt_map[ielt] == irefElt:
                nnodes_PAdapt_per_refElt_irefElt = nnodes_PAdapt_in_element[ielt]
                if fig == 1:
                    uElt = U[count:count+nnodes_PAdapt_per_refElt_irefElt] 
                elif fig == 2:
                    uElt = U2[count:count+nnodes_PAdapt_per_refElt_irefElt]   
                elif fig == 3:
                    uElt = U[count:count+nnodes_PAdapt_per_refElt_irefElt] - U2[count:count+nnodes_PAdapt_per_refElt_irefElt] 
                else :
                    raise Exception('Fig not found')       
                u.extend(uElt)
                for idim in range(ndim):    
                    dims[idim].XElt_padapt = dims[idim].X_padapt[count:count+nnodes_PAdapt_per_refElt_irefElt,:]
                    dims[idim].x_padapt.extend(dims[idim].XElt_padapt)   
                refEltType_irefElt = eltType[ielt] 
            count = count + nnodes_PAdapt_in_element[ielt]    
        refEl = cre.createReferenceElement(refEltType_irefElt,nnodes_PAdapt_per_refElt_irefElt) 
        nelts_listDegRef = refElts_deg[irefElt-1] 
        nelts_listDegRef = float(nelts_listDegRef)
        x_padapt=[]
        for idim in range(ndim):
            x_padapt.extend(dims[idim].x_padapt)    
        u = np.transpose(np.matrix(u))   
        if max(u) > Max:
            Max = max(u)
        if min(u) < Min:
            Min = min(u)
    return Min, Max  
#*******************************************************************************************************************************                   



#*******************************************************************************************************************************
# Plot solutions                                                                                                               *
#*******************************************************************************************************************************

#***********************************************************************************************************
# Fig 1                                                                                                    *
#***********************************************************************************************************
Min, Max = boundariesColorBar(1)
refEltType_irefElt = None
nnodes_PAdapt_per_refElt_irefElt = None
class Dim:
    X_padapt    = np.matrix(np.zeros((nodes_PAdapt.shape[0],1)))
    x_padapt    = [] 
    XElt_padapt = [] 
plt.figure()
plt.figure(1)
for irefElt in range (1,NrefElts+1):
    X_mesh = nodes
    dims = [] 
    for idim in range(ndim):
        dim = Dim()
        dim.X_padapt = nodes_PAdapt[:,idim] 
        dims.append(dim)   
    u = []
    for idim in range(ndim):
        dims[idim].x_padapt = [] 
    count = 0
    for ielt in range(sum(Nelts_per_refElt)):
        if elts_refElt_map[ielt] == irefElt:
            nnodes_PAdapt_per_refElt_irefElt = nnodes_PAdapt_in_element[ielt]
            uElt = U[count:count+nnodes_PAdapt_per_refElt_irefElt] 
            u.extend(uElt)
            for idim in range(ndim):    
                dims[idim].XElt_padapt = dims[idim].X_padapt[count:count+nnodes_PAdapt_per_refElt_irefElt,:]
                dims[idim].x_padapt.extend(dims[idim].XElt_padapt)   
            refEltType_irefElt = eltType[ielt] 
        count = count + nnodes_PAdapt_in_element[ielt]    
    refEl = cre.createReferenceElement(refEltType_irefElt,nnodes_PAdapt_per_refElt_irefElt) 
    nelts_listDegRef = refElts_deg[irefElt-1] 
    nelts_listDegRef = float(nelts_listDegRef)
    cont = 0
    logar = 0
    plotSuper = 0
    cbaxis = [[0, 1.]]
    x_padapt=[]
    
    for idim in range(ndim):
        x_padapt.extend(dims[idim].x_padapt) 
 
    u = np.transpose(np.matrix(u))   
    plotSolution_PAdapt(X_mesh, x_padapt, ndim, nnodesAllEltsRefElts[irefElt-1], Nelts_per_refElt[irefElt-1],
    u, refEl, nelts_listDegRef, cont, logar, plotSuper, cbaxis , Min, Max)  
    if irefElt == 1:
        plt.colorbar()

#***********************************************************************************************************
# Fig 2                                                                                                    *
#***********************************************************************************************************
Min, Max = boundariesColorBar(2)
refEltType_irefElt = None
nnodes_PAdapt_per_refElt_irefElt = None
class Dim:
    X_padapt    = np.matrix(np.zeros((nodes_PAdapt.shape[0],1)))
    x_padapt    = [] 
    XElt_padapt = [] 
plt.figure()
plt.figure(2)
for irefElt in range (1,NrefElts+1):
    X_mesh = nodes
    dims = [] 
    for idim in range(ndim):
        dim = Dim()
        dim.X_padapt = nodes_PAdapt[:,idim] 
        dims.append(dim)   
    u = []
    for idim in range(ndim):
        dims[idim].x_padapt = [] 
    count = 0
    for ielt in range(sum(Nelts_per_refElt)):
        if elts_refElt_map[ielt] == irefElt:
            nnodes_PAdapt_per_refElt_irefElt = nnodes_PAdapt_in_element[ielt]
            uElt = U2[count:count+nnodes_PAdapt_per_refElt_irefElt] 
            u.extend(uElt)
            for idim in range(ndim):    
                dims[idim].XElt_padapt = dims[idim].X_padapt[count:count+nnodes_PAdapt_per_refElt_irefElt,:]
                dims[idim].x_padapt.extend(dims[idim].XElt_padapt)   
            refEltType_irefElt = eltType[ielt] 
        count = count + nnodes_PAdapt_in_element[ielt]    
    refEl = cre.createReferenceElement(refEltType_irefElt,nnodes_PAdapt_per_refElt_irefElt) 
    nelts_listDegRef = refElts_deg[irefElt-1] 
    nelts_listDegRef = float(nelts_listDegRef)
    cont = 0
    logar = 0
    plotSuper = 0
    cbaxis = [[0, 1.]]
    x_padapt=[]
    
    for idim in range(ndim):
        x_padapt.extend(dims[idim].x_padapt) 
 
    u = np.transpose(np.matrix(u))
       
    plotSolution_PAdapt(X_mesh, x_padapt, ndim, nnodesAllEltsRefElts[irefElt-1], Nelts_per_refElt[irefElt-1],
    u, refEl, nelts_listDegRef, cont, logar, plotSuper, cbaxis, Min, Max)  
    if irefElt == 1:
        plt.colorbar()        

#***********************************************************************************************************
# Fig 3                                                                                                    *
#***********************************************************************************************************
Min, Max = boundariesColorBar(3)
refEltType_irefElt = None
nnodes_PAdapt_per_refElt_irefElt = None
class Dim:
    X_padapt    = np.matrix(np.zeros((nodes_PAdapt.shape[0],1)))
    x_padapt    = [] 
    XElt_padapt = [] 
plt.figure()
plt.figure(3)
for irefElt in range (1,NrefElts+1):
    X_mesh = nodes
    dims = [] 
    for idim in range(ndim):
        dim = Dim()
        dim.X_padapt = nodes_PAdapt[:,idim] 
        dims.append(dim)   
    u = []
    for idim in range(ndim):
        dims[idim].x_padapt = [] 
    count = 0
    for ielt in range(sum(Nelts_per_refElt)):
        if elts_refElt_map[ielt] == irefElt:
            nnodes_PAdapt_per_refElt_irefElt = nnodes_PAdapt_in_element[ielt]
            uElt = U[count:count+nnodes_PAdapt_per_refElt_irefElt]-U2[count:count+nnodes_PAdapt_per_refElt_irefElt] 
            u.extend(uElt)
            for idim in range(ndim):    
                dims[idim].XElt_padapt = dims[idim].X_padapt[count:count+nnodes_PAdapt_per_refElt_irefElt,:]
                dims[idim].x_padapt.extend(dims[idim].XElt_padapt)   
            refEltType_irefElt = eltType[ielt] 
        count = count + nnodes_PAdapt_in_element[ielt]    
    refEl = cre.createReferenceElement(refEltType_irefElt,nnodes_PAdapt_per_refElt_irefElt) 
    nelts_listDegRef = refElts_deg[irefElt-1] 
    nelts_listDegRef = float(nelts_listDegRef)
    cont = 0
    logar = 0
    plotSuper = 0
    cbaxis = [[0, 1.]]
    x_padapt=[]
    
    for idim in range(ndim):
        x_padapt.extend(dims[idim].x_padapt) 
 
    u = np.transpose(np.matrix(u))
       
    plotSolution_PAdapt(X_mesh, x_padapt, ndim, nnodesAllEltsRefElts[irefElt-1], Nelts_per_refElt[irefElt-1],
    u, refEl, nelts_listDegRef, cont, logar, plotSuper, cbaxis, Min, Max)  
    if irefElt == 1:
        plt.colorbar()

print(Fore.CYAN + "\033[1m" + '#*********************************************************************************' + "\033[0m")                                                                                                 
plt.show()
#*****************************************************************************************                       
                        


        
            

            
    






