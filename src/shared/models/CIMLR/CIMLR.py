import numpy as np
import time
from .utils import multipleK, Network_Diffusion, NE_dn, eig1, L2_distance_1, projsplx_c, umkl_bo, Kbeta, tsne_p_bo, litekmeans, dist2

def CIMLR(alldata, c, k=10):
    NITER = 30
    num = alldata[0].shape[0]
    r = -1
    beta = 0.8
    
    D_Kernels = []
    for i in range(len(alldata)):
        D_Kernels.append(multipleK(alldata[i]))
    D_Kernels = np.concatenate(D_Kernels, axis=2)
    
    alphaK = np.ones(D_Kernels.shape[2]) / D_Kernels.shape[2]
    distX = np.mean(D_Kernels, axis=2)
    distX1 = np.sort(distX, axis=1)
    idx = np.argsort(distX, axis=1, kind='stable')
    
    A = np.zeros((num, num))
    di = distX1[:, 1:k+2]
    rr = 0.5 * (k * di[:, k] - np.sum(di[:, :k], axis=1))
    id_ = idx[:, 1:k+2]
    
    temp = (di[:, k][:, None] - di) / (k * di[:, k] - np.sum(di[:, :k], axis=1) + np.finfo(float).eps)[:, None]
    a = np.repeat(np.arange(num)[:, None], id_.shape[1], axis=1)
    A[a.ravel(), id_.ravel()] = temp.ravel()
    
    if r <= 0:
        r = np.mean(rr)
    lambda_val = max(np.mean(rr), 0)
    A[np.isnan(A)] = 0
    
    S0 = np.max(distX) - distX
    S0 = Network_Diffusion(S0, k)
    S0 = NE_dn(S0, 'ave')
    S = (S0 + S0.T) / 2.0
    
    # In MATLAB order=2, sum(S,2) means sum across columns
    D0 = np.diag(np.sum(S, axis=1))
    L0 = D0 - S
    F, temp_val, evs_init = eig1(L0, c, 0)
    F = NE_dn(F, 'ave')
    
    evs = np.zeros((c, NITER + 1))
    evs[:, 0] = evs_init[:c]
    converge = np.zeros(NITER)
    S_old = np.copy(S)
    
    for iter in range(NITER):
        distf = L2_distance_1(F.T, F.T)
        A = np.zeros((num, num))
        b = idx[:, 1:]
        a = np.repeat(np.arange(num)[:, None], b.shape[1], axis=1)
        
        inda = (a.ravel(), b.ravel())
        ad = (distX[inda] + lambda_val * distf[inda]) / (2 * r)
        ad = ad.reshape((num, b.shape[1]))
        ad = projsplx_c(-ad.T).T
        A[inda] = ad.ravel()
        A[np.isnan(A)] = 0
        
        S = (1 - beta) * A + beta * S
        S = Network_Diffusion(S, k)
        S = (S + S.T) / 2.0
        
        D_mat = np.diag(np.sum(S, axis=1))
        L = D_mat - S
        F_old = F
        F, temp_val, ev = eig1(L, c, 0)
        F = NE_dn(F, 'ave')
        F = (1 - beta) * F_old + beta * F
        evs[:, iter+1] = ev[:c]
        
        DD = np.zeros(D_Kernels.shape[2])
        for i in range(D_Kernels.shape[2]):
            temp_DD = (np.finfo(float).eps + D_Kernels[:, :, i]) * (np.finfo(float).eps + S)
            DD[i] = np.mean(np.sum(temp_DD, axis=0))
            
        alphaK0 = umkl_bo(DD)
        alphaK0 = alphaK0 / np.sum(alphaK0)
        alphaK = (1 - beta) * alphaK + beta * alphaK0
        alphaK = alphaK / np.sum(alphaK)
        
        fn1 = np.sum(ev[:c])
        fn2 = np.sum(ev[:c+1]) if len(ev) > c else fn1
        converge[iter] = fn2 - fn1
        
        if iter < 9:
            if ev[-1] > 0.000001:
                lambda_val = 1.5 * lambda_val
                r = r / 1.01
        else:
            if converge[iter] > 1.01 * converge[iter-1]:
                S = S_old
                if converge[iter-1] > 0.2:
                    print('Warning: Maybe you should set a larger value of c')
                break
        
        S_old = np.copy(S)
        distX = Kbeta(D_Kernels, alphaK)
        idx = np.argsort(distX, axis=1, kind='stable')
        
    LF = F
    return S, LF, alphaK, converge
