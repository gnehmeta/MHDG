#*********************************************************************************************************************************************
# Modules                                                                                                                                    *
#*********************************************************************************************************************************************
import HDF5load as h
import numpy as npy
import math as m
import cons2phys as c2p
import setSubPlot as sp
import createReferenceElement as cre
import matplotlib.pyplot as plt
import plotMesh as pm
import plotSolution as ps
from colorama import Fore
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# Parallel/serial                                                                                                                            *
#*********************************************************************************************************************************************
parallel = 0 # true = 1
nproc    = 1 # number of MPI task for parallel (mesh separation)
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# Plot options                                                                                                                               *
#*********************************************************************************************************************************************
plotPhys = 1 # Plot physical values (true=1)
plotCons = 0 # Plot conservative variables
# Dimensional (1) or non-dimensional (0) plots
phys_dimensional_plots = 1 # physical variables
cons_dimensional_plots = 0 # conservative variables
nref = 3 # plot order
startPlot = 1 #default: 1 Starting number for plot (useful if no close all)
gatherPlot = 0 #True to gather everything in one figure
cBound = 0 #True: bound colorbar axis (to adapt the boundaries see below)
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# Printing out                                                                                                                               *
#*********************************************************************************************************************************************
printOut = 0 # Print out plots
path2save = '/Home/GN272030/Bureau/Results/Img_MHDG/'
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# Solution                                                                                                                                   *
#*********************************************************************************************************************************************
solPath  = ['/Home/GN272030/Bureau/Results/Solution/']
meshPath = ['/Home/GN272030/Bureau/Results/Mesh/']
#solName  = 'Sol2D_West_NoHole_Nel3895_P4_DPe0.260E+02_DPai0.314E+06_DPae0.105E+08'
#solName = 'Sol2D_CircLimAlign_Quads_Nel208_P4_DPe0.200E+02_DPai0.314E+06_DPae0.105E+08_0001'
solName  = 'solInit'
solName1 = 'solFin'
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# Visualisation Solution EBC                                                                                                                 *                 *
#*********************************************************************************************************************************************
if parallel:
    solNameComp = [solName, '_', str(1), '_', str(nproc), '.h5']
else:
    solNameComp  = [solName , '.h5']
    solNameComp1 = [solName1, '.h5']
Data  = h.HDF5load(' '.join(map(str,solPath+solNameComp )).replace(" ", ""))
Data1 = h.HDF5load(' '.join(map(str,solPath+solNameComp1)).replace(" ", ""))
if 'nnodes_PAdapt_in_element' in Data:
    import visualisation_sol_EBC
else: 
    #*****************************************************************************************************************************************
    # Initial message                                                                                                                        *
    #*****************************************************************************************************************************************   
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
    #*****************************************************************************************************************************************



    #*****************************************************************************************************************************************
    # Bound axis                                                                                                                             *
    #*****************************************************************************************************************************************
    if plotPhys and cBound:
        if phys_dimensional_plots:
            # rho, u, Ei, Ee, Pi, Pe, Ti, Te, Csi, M , rhon
            cpaxis= npy.matrix([[0, 1.2e19], [-0.25, 0.25], [0, 3.5e9], [0, 3.5e9],
                                [0, 2.3e28], [0, 2.3e28], [0, 50], [0, 50], [2, 7e4],
                                [-1.1, 1.1], [0, 4.0e-3]])
        else:
            # rho, u, Ei, Ee, Pi, Pe, Ti, Te, Csi, M , rhon
            cpaxis= npy.matrix([[0, 1.2], [-0.25, 0.25], [0, 20], [0, 20], [0, 1], [0, 1],
                                [0, 1],[0, 1], [1, 5], [-1.1, 1.1], [0, 4.0e-3]])
    else:
        cpaxis= npy.matrix([[0, 1.2], [-0.25, 0.25], [0, 20], [0, 20], [0, 1], [0, 1], [0, 1],
                            [0, 1], [1, 5], [-1.1, 1.1], [0, 4.0e-3]])

    if plotCons and cBound:
        if cons_dimensional_plots:
            # rho, Gamma (Nu), NEi, NEe, rhon
            ccaxis= [[0, 1.2e19], [-1.6e16, 1.6e16], [0, 3.5e28], [0, 3.5e28], [0, 3.0e-3]]
        else:
            ccaxis= [[0, 1.2], [-0.9, 0.9], [0, 20], [0, 20], [0, 3.0e-3]]
    else:
        ccaxis= [[0, 1.2], [-0.9, 0.9], [0, 20], [0, 20], [0, 3.0e-3]]
    #*****************************************************************************************************************************************    


    #*****************************************************************************************************************************************
    # Working...                                                                                                                             *
    #*****************************************************************************************************************************************
    pos = solName.find('_P')
    for i in range(1,11):
        if solName[pos+i] == '_' :
            pos = pos+i
            break
    if not parallel:
        nproc = 1
    #*****************************************************************************************************************************************    


    #*****************************************************************************************************************************************
    # Definition                                                                                                                             *
    #*****************************************************************************************************************************************
    def boundariesColorBar():
        Max = [0]*11 
        Min = [10**100]*11 
        for iproc in range (1,nproc+1):
            if parallel:
                meshName = [solName[6:pos], '_', str(iproc), '_', str(nproc), '.h5']
                if solName[3:5]=='3D':
                    solNameComp = [solName, '_ip', str(iproc), '_it1_np', str(nproc), '.h5']
                else:
                    solNameComp = [solName, '_', str(iproc), '_', str(nproc), '.h5']
            else:
                meshName = [solName[6:pos], '.h5']
                solNameComp = [solName, '.h5'] 
            Data = h.HDF5load(' '.join(map(str,solPath+solNameComp)).replace(" ", ""))
            Mesh = h.HDF5load(' '.join(map(str,meshPath+meshName)).replace(" ", ""))
            Mesh['lscale'] = Data['length_scale']
            Neq = Data['Neq'][0]
            u = Data['u']
            u = npy.transpose(npy.matrix(u))
            uNbElement = u.shape[0]
            uc = npy.matrix(npy.zeros((m.floor(uNbElement/Neq),Neq)))
            for i in range(m.floor(uNbElement/Neq)):
                uc[i,:] = npy.transpose(u[i*Neq:(i+1)*Neq,0])  
            model = Data['model'][0].decode('UTF-8')
            a = Data['a'][0]
            Mref = Data['Mref'][0]
            reference_values_physical_variables = Data['reference_values_physical_variables'][:] 
            physical_variable_names = Data['physical_variable_names'][:]    
            up = c2p.cons2phys(uc,model,a,Mref) 
            
            for ii in range(up.shape[1]):
                uplot = up[:,ii]
                if phys_dimensional_plots:
                    uplot = up[:,ii]  * reference_values_physical_variables[ii]  
                if max(uplot) > Max[ii]:
                    Max[ii] = max(uplot)
                if min(uplot) < Min[ii]:
                    Min[ii] = min(uplot)
        return Min,Max   
    #*****************************************************************************************************************************************                   



    #*****************************************************************************************************************************************
    # Plot solutions                                                                                                                         *
    #*****************************************************************************************************************************************
    Min,Max = boundariesColorBar()

    for iproc in range (1,nproc+1):
        # Load results
        if parallel:
            meshName = [solName[6:pos], '_', str(iproc), '_', str(nproc), '.h5']
            if solName[3:5]=='3D':
                solNameComp = [solName, '_ip', str(iproc), '_it1_np', str(nproc), '.h5']
            else:
                solNameComp = [solName, '_', str(iproc), '_', str(nproc), '.h5']
            
        else:
            meshName = [solName[6:pos], '.h5']
            solNameComp = [solName, '.h5']
        Data = h.HDF5load(' '.join(map(str,solPath+solNameComp)).replace(" ", ""))
        Mesh = h.HDF5load(' '.join(map(str,meshPath+meshName)).replace(" ", ""))	
        Mesh['lscale'] = Data['length_scale']
        
        #Number of equations
        Neq = Data['Neq'][0]
        
        #Conservative and physical variables
        u = Data['u']
        u = npy.transpose(npy.matrix(u))
        uNbElement = u.shape[0]
        uc = npy.matrix(npy.zeros((m.floor(uNbElement/Neq),Neq)))
        for i in range(m.floor(uNbElement/Neq)):
            uc[i,:] = npy.transpose(u[i*Neq:(i+1)*Neq,0])  
        model = Data['model'][0].decode('UTF-8')
        a = Data['a'][0]
        Mref = Data['Mref'][0]
        reference_values_physical_variables = Data['reference_values_physical_variables'][:] 
        physical_variable_names = Data['physical_variable_names'][:]    
        up = c2p.cons2phys(uc,model,a,Mref)
        nc = uc.shape[1]+1
        np = up.shape[1]+1
        [nrowc, ncolc, nrowp, ncolp] = sp.setSubPlot(model)
        
        # Reference element on plane
        if Mesh['elemType'][0] == 1:
            elemType=0
        elif Mesh['elemType'][0]==0:
            elemType=1
        Mesh['T'] = npy.transpose(Mesh['T'])   
        refEl = cre.createReferenceElement(elemType,Mesh['T'].shape[1])
        
        if cBound:
            cont = 2
        else:
            cont = 0
        
        if solName[3:5] == '2D' :
            if iproc == 1:
                plt.figure()
            plt.figure(1)
            plt.title('Mesh')
            pm.plotMesh(Mesh['X'],Mesh['T'],elemType)  
            plt.axis('on')    
            plt.axis('equal') 
            
            if plotPhys:
                iPlot = startPlot + 1
                for ii in range(up.shape[1]):
                    uplot = up[:,ii]
                    if phys_dimensional_plots:
                        uplot = up[:,ii]  * reference_values_physical_variables[ii]

                    name = str(physical_variable_names[ii])
                    name = name.replace("b","")
                    name = name.replace("'","")

                    if iproc == 1:
                        plt.figure()          

                    if name == 'rhon':
                        plt.figure(iPlot)
                        plt.title(name)
                        ps.plotSolution(Mesh['X'],Mesh['lscale'],Mesh['T'],
                        npy.absolute(uplot),refEl,nref,cont,1,0,cpaxis[ii,:], Max[ii], Min[ii])
                        if not phys_dimensional_plots :
                            caxis(np.array([- 4,0]))
                        if iproc == 1:
                            plt.colorbar()
                        plt.axis('off')
                        iPlot = iPlot +1     
                    else:
                        plt.figure(iPlot)
                        plt.title(name)      
                        ps.plotSolution(Mesh['X'],Mesh['lscale'],Mesh['T'],uplot,refEl,
                        nref,cont,0,0,cpaxis[ii,:],Max[ii], Min[ii])
                        if iproc == 1:
                            plt.colorbar()
                        plt.axis('off')
                        iPlot = iPlot +1
                    percentage = m.floor(100*(ii+1+up.shape[1]*(iproc-1))/(nproc*up.shape[1]))    
                    if percentage == 100: 
                        print(Fore.GREEN + "\033[1m" + '[COMPLETED!]' + "\033[0m")
                    else:           
                        print(Fore.CYAN +"\033[1m" + f'[Processing...' + "\033[0m", percentage, '%'+Fore.CYAN + "\033[1m" +']'+ "\033[0m",sep="") 
    print(Fore.CYAN + "\033[1m" + '#*********************************************************************************' + "\033[0m")                                                                                                 
    plt.show()
    #*****************************************************************************************************************************************                      
                        


        
            

            
    






