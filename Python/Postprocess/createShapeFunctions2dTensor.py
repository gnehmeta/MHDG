#****************************************************************************************************
# Modules                                                                                           * 
#****************************************************************************************************
import numpy as np
import numpy.matlib
#****************************************************************************************************



#****************************************************************************************************
# Definition 1                                                                                      *
#****************************************************************************************************
def bsxfunTimes(X,Y):
    Z = np.matrix(np.zeros((X.shape[0],Y.shape[1])))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = X[i,0]*Y[0,j]
    return Z
#****************************************************************************************************



#****************************************************************************************************
# Definition 2                                                                                      *
#****************************************************************************************************
def colsToCol(X):
    X = np.matrix(X)
    Y = np.matrix(np.zeros((X.shape[0]*X.shape[1] ,1)))
    for i in range(X.shape[1]):
        Y[i*X.shape[0]:(i+1)*X.shape[0]] = X[:,i]     
    return Y     
#****************************************************************************************************



#****************************************************************************************************
# Definition 3                                                                                      *
#****************************************************************************************************
def createShapeFunctions2dTensor(shapeFun1d = None,gw1d = None,gp1d = None,perm = None): 
    n1d = shapeFun1d.shape[1]
    ng = shapeFun1d.shape[2]
    ind_x = np.reshape(gp1d, tuple(np.array([ng,1])), order="F")
    ind_y = np.reshape(gp1d, tuple(np.array([1,ng])), order="F")

    gp2d1Mat = np.matrix(np.matlib.repmat(ind_x,1,np.asarray(gp1d).size))
    gp2d1Vec = np.matrix(np.zeros((gp2d1Mat.shape[0]*gp2d1Mat.shape[1],1)))
    gp2d2Mat = np.transpose(np.matrix(np.matlib.repmat(np.transpose(ind_y),1,np.asarray(gp1d).size)))
    gp2d2Vec = np.matrix(np.zeros((gp2d2Mat.shape[0]*gp2d2Mat.shape[1],1)))
    for i in range(gp2d1Mat.shape[1]):
        gp2d1Vec[i*gp2d1Mat.shape[0]:(i+1)*gp2d1Mat.shape[0]] = gp2d1Mat[:,i]

    for i in range(gp2d2Mat.shape[1]):  
        gp2d2Vec[i*gp2d2Mat.shape[0]:(i+1)*gp2d2Mat.shape[0]] = gp2d2Mat[:,i]      
        
    
    gp2d = np.matrix(np.zeros((gp2d1Vec.shape[0],2))) 
    for i in range(gp2d1Vec.shape[0]):
        gp2d[i,0] = gp2d1Vec[i,0]
        gp2d[i,1] = gp2d2Vec[i,0]

    gw1d_i = np.reshape(gw1d, tuple(np.array([ng,1])), order="F")
    gw1d_j = np.reshape(gw1d, tuple(np.array([1,ng])), order="F")
    gw2d = bsxfunTimes(gw1d_i,gw1d_j)
    
    shapeFun1d_i = np.zeros((n1d, 1, ng, 1, 2))
    for k in range(2):
        for j in range(ng):
            shapeFun1d_i[:,0,j,0,k] = shapeFun1d[k,:,j]

    shapeFun1d_j = np.zeros((1, n1d, 1, ng, 2))
    for k in range(2):
        for j in range(ng):
            shapeFun1d_j[0,:,0,j,k] = np.transpose(shapeFun1d[k,:,j])  

     
    shapeFun1d_i1= np.zeros((ng,n1d,1))       
    for i in range(ng):
        for j in range(n1d):
            shapeFun1d_i1[i,j,0] = shapeFun1d_i[j,0,i,:,0] 
    
    shapeFun1d_i2= np.zeros((ng,n1d,1))       
    for i in range(ng):
        for j in range(n1d):
            shapeFun1d_i2[i,j,0] = shapeFun1d_i[j,0,i,:,1]

    shapeFun1d_j1= np.zeros((ng,1,1,n1d))       
    for i in range(ng):
        for j in range(n1d):
            shapeFun1d_j1[i, 0 ,:,j] = shapeFun1d_i[j,0,i,:,0]

    shapeFun1d_j2= np.zeros((ng,1,1,n1d))      
    for i in range(ng):
        for j in range(n1d):
            shapeFun1d_j2[i, 0 ,:,j] = shapeFun1d_i[j,0,i,:,1]        

    shapeFun2d_1 = np.zeros((ng**2,n1d,n1d))
    k = 0
    for i in range(ng):
        for j in range(ng):
            shapeFun2d_1[k,:,:] = np.transpose(bsxfunTimes(shapeFun1d_i1[i,:,:],shapeFun1d_j1[j,0,:,:])) 
            k = k+1        

    shapeFun2d_2 = np.zeros((ng**2,n1d,n1d))
    k = 0
    for i in range(ng):
        for j in range(ng):
            shapeFun2d_2[k,:,:] = bsxfunTimes(shapeFun1d_i2[i,:,:],shapeFun1d_j1[j,0,:,:])
            k = k+1   

    shapeFun2d_3 = np.zeros((ng**2,n1d,n1d))
    k = 0
    for i in range(ng):
        for j in range(ng):
            shapeFun2d_3[k,:,:] = bsxfunTimes(shapeFun1d_i1[i,:,:],shapeFun1d_j2[j,0,:,:])
            k = k+1                        

    shapeFun2d_vec = np.zeros((3*ng**2,n1d,n1d))
    for i in range(3*ng**2):
        if i<= ng**2-1:
            shapeFun2d_vec[i,:,:] = shapeFun2d_1[i,:,:] 
        elif i >= ng**2 and i <= 2*ng**2-1:
            shapeFun2d_vec[i,:,:] = shapeFun2d_2[i-ng**2,:,:]
        else:
            shapeFun2d_vec[i,:,:] = shapeFun2d_3[i-2*ng**2,:,:]         
              
    shapeFun2d = np.zeros((3,n1d**2,ng**2))
    for i in range(3):
        shapeFun2d[i,:,:] = np.matrix(np.zeros((n1d**2,ng**2)))
        for j in range(ng**2):
            shapeFun2d[i,:,j] = colsToCol(shapeFun2d_vec[j,:,:]).reshape((1,-1))

    shapeFun2dPerm = np.zeros((3,n1d**2,ng**2)) 
    for j in range(3):
        for i in range(perm.shape[0]):
            shapeFun2dPerm[j,perm[i]-1,:] = shapeFun2d[j,i,:] 

    shapeFun2d = shapeFun2dPerm
    return shapeFun2d,gw2d,gp2d
    #****************************************************************************************************
    


    