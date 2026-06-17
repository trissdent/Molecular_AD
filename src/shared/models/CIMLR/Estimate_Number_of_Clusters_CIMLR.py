import numpy as np
from utils import multipleK, Network_Diffusion
from scipy.sparse import spdiags

def discretisationEigenVectorData(EigenVector):
    n, k = EigenVector.shape
    J = np.argmax(EigenVector, axis=1)
    Y = np.zeros((n, k))
    Y[np.arange(n), J] = 1
    return Y

def discretisation(EigenVectors):
    n, k = EigenVectors.shape
    vm = np.sqrt(np.sum(EigenVectors**2, axis=1))
    EigenVectors = EigenVectors / (vm[:, None] + np.finfo(float).eps)
    
    R = np.zeros((k, k))
    R[:, 0] = EigenVectors[int(n/2), :]
    c = np.zeros(n)
    for j in range(1, k):
        c = c + np.abs(EigenVectors @ R[:, j-1])
        i = np.argmin(c)
        R[:, j] = EigenVectors[i, :]
        
    lastObjectiveValue = 0
    exitLoop = 0
    nbIterationsDiscretisation = 0
    nbIterationsDiscretisationMax = 20
    
    while exitLoop == 0:
        nbIterationsDiscretisation += 1
        EigenvectorsDiscrete = discretisationEigenVectorData(EigenVectors @ R)
        U, S_vals, Vt = np.linalg.svd(EigenvectorsDiscrete.T @ EigenVectors + np.finfo(float).eps, full_matrices=False)
        NcutValue = 2 * (n - np.sum(S_vals))
        
        if abs(NcutValue - lastObjectiveValue) < np.finfo(float).eps or nbIterationsDiscretisation > nbIterationsDiscretisationMax:
            exitLoop = 1
        else:
            lastObjectiveValue = NcutValue
            R = Vt.T @ U.T
            
    return EigenvectorsDiscrete, EigenVectors

def Estimate_Number_of_Clusters_given_graph(W, NUMC):
    NUMC = np.array(NUMC)
    if np.min(NUMC) == 1:
        print('Warning: Note that we always assume there are more than one cluster.')
        NUMC = NUMC[NUMC > 1]
        
    W = (W + W.T) / 2.0
    degs = np.sum(W, axis=1)
    D = np.diag(degs)
    L = D - W
    degs[degs == 0] = np.finfo(float).eps
    D_inv_sqrt = np.diag(1.0 / (degs**0.5))
    L = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalue, U = np.linalg.eig(L)
    eigenvalue = np.real(eigenvalue)
    U = np.real(U)
    idx = np.argsort(eigenvalue)
    eigenvalue = eigenvalue[idx]
    U = U[:, idx]
    
    quality = np.zeros(len(NUMC))
    for ck_idx, ck in enumerate(NUMC):
        if ck == 1:
            quality[ck_idx] = np.sum(np.diag(1.0 / (U[:, 0] + np.finfo(float).eps)) @ U[:, 0])
        else:
            UU = U[:, :ck]
            UU = UU / (np.sqrt(np.sum(UU**2, axis=1))[:, None] + np.finfo(float).eps)
            EigenvectorsDiscrete, EigenVectors = discretisation(UU)
            EigenVectors = EigenVectors**2
            temp1 = np.sort(EigenVectors, axis=1)[:, ::-1]
            val = (1 - eigenvalue[ck]) / (1 - eigenvalue[ck-1]) * np.sum(np.diag(1.0 / (temp1[:, 0] + np.finfo(float).eps)) @ temp1[:, :max(2, ck-1)])
            quality[ck_idx] = val
            
    return quality

def Estimate_Number_of_Clusters_CIMLR(alldata, NUMC):
    NUMC = np.array(NUMC)
    W = None
    for i in range(len(alldata)):
        D_Kernels = multipleK(alldata[i])
        distX = np.mean(D_Kernels, axis=2)
        W0 = np.max(distX) - distX
        k_val = max(int(np.ceil(alldata[0].shape[0]/20)), 10)
        
        if i == 0:
            W = Network_Diffusion(W0, k_val)
        else:
            W = W + Network_Diffusion(W0, k_val)
            
    Quality = Estimate_Number_of_Clusters_given_graph(W, NUMC)
    Quality_plus = Estimate_Number_of_Clusters_given_graph(W, NUMC + 1)
    Quality_minus = Estimate_Number_of_Clusters_given_graph(W, NUMC - 1)
    
    K1 = 2 * (1 + Quality) - (2 + Quality_plus + Quality_minus)
    K2 = K1 * (NUMC + 1) / NUMC
    
    return K1, K2
