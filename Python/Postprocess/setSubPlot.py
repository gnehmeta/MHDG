#****************************************************************************************************
# Definition                                                                                        *
#****************************************************************************************************
def setSubPlot(model):

	if model == 'N-Gamma':
		nrowc = 1
		ncolc = 3
		nrowp = 1
		ncolp = 3
	elif model == 'N-Gamma-Neutral':
		nrowc = 1
		ncolc = 4
		nrowp = 1
		ncolp = 4
	elif model =='N-Gamma-Ti-Te':
		nrowc = 1
		ncolc = 5
		nrowp = 2
		ncolp = 6
	elif model == 'N-Gamma-Ti-Te-Neutral':
		nrowc = 1
		ncolc = 6
		nrowp = 2
		ncolp = 6
	elif model == 'N-Gamma-Vorticity':
		nrowc = 1
		ncolc = 5
		nrowp = 1
		ncolp = 5
	return nrowc,ncolc,nrowp,ncolp
#****************************************************************************************************	
