#************************************************************************************************		
# Modules																	 			        *
#************************************************************************************************
import numpy as np
#************************************************************************************************



#************************************************************************************************
# Definition															     				    *
#************************************************************************************************
def cons2phys(uc,model,a,Mref):
	if model == 'N-Gamma':
		up = np.matrix(np.zeros((uc.shape[0],2)))
		for i in range(up.shape[0]): 
			up[i,0] = uc[i,0]                                       # density
			up[i,1] = (uc[i,1]/uc[i,0])/np.sqrt(a)                  # Mach
	elif model == 'N-Gamma-Neutral':
		up = np.zeros((uc.shape[0],4))
		for i in range(up.shape[0]): 
			up[i,0] = uc[i,0]                                       # density
			up[i,1] = uc[i,1]/uc[i,0]                               # u
			up[i,2] = (uc[i,1]/uc[i,0])/np.sqrt(a)                  # Mach
			up[i,3] = uc[i,2]                                       # neutral
	elif model =='N-Gamma-Ti-Te': 
		up = np.matrix(np.zeros((uc.shape[0],10)))
		up[:,0] = uc[:,0] 
		for i in range(up.shape[0]):                                # density
			up[i,1] = uc[i,1]/uc[i,0]                               # u
			up[i,2] = uc[i,2]/uc[i,0]                               # total energy for ions
			up[i,3] = uc[i,3]/uc[i,0]                               # total energy for electrons
			up[i,4] = (2/(3*Mref)*(uc[i,2]-0.5*uc[i,1]**2/uc[i,0])) # pressure for ions
			up[i,5] = (2/(3*Mref)*uc[i,3])                          # pressure for electrons
			up[i,6] = up[i,4]/up[i,0]                               # temperature of ions
			up[i,7] = up[i,5]/up[i,0]                               # temperature of electrons
			up[i,8] = np.sqrt((abs(up[i,6])+abs(up[i,7]))*Mref)     # sound speed
			up[i,9] = up[i,1]/up[i,8]                               # Mach   
	elif model == 'N-Gamma-Ti-Te-Neutral':
		up = np.matrix(np.zeros((uc.shape[0],11)))
		for i in range(up.shape[0]): 
			up[i,0] = uc[i,0]                                       # density
			up[i,1] = uc[i,1]/uc[i,0]                               # u
			up[i,2] = uc[i,2]/uc[i,0]                               # total energy for ions
			up[i,3] = uc[i,3]/uc[i,0]                               # total energy for electrons
			up[i,4] = (2/(3*Mref)*(uc[i,2]-0.5*uc[i,1]**2/uc[i,0])) # pressure for ions
			up[i,5] = (2/(3*Mref)*uc[i,3])                          # pressure for electrons
			up[i,6] = up[i,4]/up[i,0]                               # temperature of ions
			up[i,7] = up[i,5]/up[i,0]                               # temperature of electrons
			up[i,8] = np.sqrt((abs(up[i,6])+abs(up[i,7]))*Mref)     # sound speed
			up[i,9] = up[i,1]/up[i,8]                               # Mach
			up[i,10]= uc[i,4]                                       # neutral
	elif model == 'N-Gamma-Vorticity':
		up = np.matrix(np.zeros((uc.shape[0],4))) 
		for i in range(up.shape[0]): 
			up[i,0] = uc[i,0]                                       # density
			up[i,1] = uc[i,1]/uc[i,0]/np.sqrt(a)                    # Mach
			up[i,2] = uc[i,2]                                       # total energy for ions
			up[i,3] = uc[i,3]                                       # total energy for electrons    
	return up  
#************************************************************************************************	  

