import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import norm, beta
from sklearn.cluster import KMeans

def dist2(x, c=None):
    if c is None:
        c = x
    return dist.cdist(x, c, 'sqeuclidean')

def L2_distance_1(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros(a.shape[1])])
        b = np.vstack([b, np.zeros(b.shape[1])])
    
    aa = np.sum(a*a, axis=0)
    bb = np.sum(b*b, axis=0)
    ab = a.T @ b
    d = aa[:, None] + bb[None, :] - 2 * ab
    d = np.maximum(d, 0)
    np.fill_diagonal(d, 0)
    return d

def multipleK(x):
    N = x.shape[0]
    sigma = np.arange(2, 0.75, -0.25)
    Diff = dist2(x)
    T = np.sort(Diff, axis=1)
    allk = np.arange(10, 31, 2)
    Kernels = []
    
    for idx_k in range(len(allk)):
        if allk[idx_k] < (N - 1):
            TT = np.mean(T[:, 1:(allk[idx_k]+1)], axis=1) + np.finfo(float).eps
            Sig = (TT[:, None] + TT[None, :]) / 2.0
            Sig = Sig * (Sig > np.finfo(float).eps) + np.finfo(float).eps
            for j in range(len(sigma)):
                W = norm.pdf(Diff, 0, sigma[j] * Sig)
                Kernels.append((W + W.T) / 2.0)
                
    Kernels = np.stack(Kernels, axis=-1)
    D_Kernels = np.zeros_like(Kernels)
    for i in range(Kernels.shape[2]):
        K = Kernels[:, :, i]
        k_val = 1.0 / np.sqrt(np.diag(K) + 1.0)
        G = K * np.outer(k_val, k_val)
        diagG = np.diag(G)
        D_Kernels[:, :, i] = (diagG[:, None] + diagG[None, :] - 2 * G) / 2.0
        np.fill_diagonal(D_Kernels[:, :, i], 0)
        
    return D_Kernels

def dominateset(aff_matrix, NR_OF_KNN):
    A = np.sort(aff_matrix, axis=1)[:, ::-1]
    B = np.argsort(aff_matrix, axis=1)[:, ::-1]
    res = A[:, :NR_OF_KNN]
    loc = B[:, :NR_OF_KNN]
    PNN_matrix1 = np.zeros_like(aff_matrix)
    inds = np.arange(aff_matrix.shape[0])[:, None]
    PNN_matrix1[inds, loc] = res
    PNN_matrix = (PNN_matrix1 + PNN_matrix1.T) / 2.0
    return PNN_matrix

def NE_dn(w, type_='ave'):
    w = w * w.shape[0]
    D = np.sum(np.abs(w), axis=1) + np.finfo(float).eps
    if type_ == 'ave':
        D = 1.0 / D
        wn = D[:, None] * w
    elif type_ == 'gph':
        D = 1.0 / np.sqrt(D)
        wn = D[:, None] * (w * D[None, :])
    return wn

def TransitionFields(W):
    zeroindex = np.where(np.sum(W, axis=1) == 0)[0]
    W = W * W.shape[0]
    W = NE_dn(W, 'ave')
    w = np.sqrt(np.sum(np.abs(W), axis=0) + np.finfo(float).eps)
    W = W / w[None, :]
    W = W @ W.T
    Wnew = np.copy(W)
    Wnew[zeroindex, :] = 0
    Wnew[:, zeroindex] = 0
    return Wnew

def Network_Diffusion(A, K):
    np.fill_diagonal(A, 0)
    P = dominateset(np.abs(A), min(K, A.shape[0]-1)) * np.sign(A)
    DD = np.sum(np.abs(P.T), axis=0)
    P = P + np.eye(P.shape[0]) + np.diag(np.sum(np.abs(P.T), axis=0))
    P = TransitionFields(P)
    D_vals, U = np.linalg.eig(P)
    d = np.real(D_vals) + np.finfo(float).eps
    alpha = 0.8
    beta_param = 2
    d = (1 - alpha) * d / (1 - alpha * d**beta_param)
    D_mat = np.diag(np.real(d))
    W = U @ D_mat @ U.T
    np.fill_diagonal(W, 0)
    W = W / (1 - np.diag(W))[:, None]
    W = DD[:, None] * W
    W = (W + W.T) / 2.0
    return W

def eig1(A, c, isMax=1, isSym=1):
    if isSym == 1:
        A = np.maximum(A, A.T)
    d, v = np.linalg.eig(A)
    if isMax == 0:
        idx = np.argsort(d)
    else:
        idx = np.argsort(d)[::-1]
    
    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = np.real(v[:, idx1])
    eigval_full = d[idx]
    return eigvec, eigval, eigval_full

def projsplx_c(y):
    m, n = y.shape
    x = np.zeros((m, n))
    for k in range(n):
        s = y[:, k]
        means = np.sum(s)
        mins = np.min(s)
        s_shifted = s - (means - 1.0)/m
        if mins < 0:
            f = 1.0
            lambda_m = 0.0
            vs = np.zeros(m)
            ft = 1
            while abs(f) > 1e-10:
                npos = 0
                f = 0.0
                for j in range(m):
                    vs[j] = s_shifted[j] - lambda_m
                    if vs[j] > 0:
                        npos += 1
                        f += vs[j]
                if npos > 0:
                    lambda_m += (f - 1.0) / npos
                if ft > 100:
                    x[:, k] = np.maximum(vs, 0)
                    break
                ft += 1
            if ft <= 100:
                x[:, k] = np.maximum(vs, 0)
        else:
            x[:, k] = s_shifted
    return x

def Hbeta(D, beta_val):
    D_min = np.min(D)
    D_max = np.max(D)
    D_scaled = (D - D_min) / (D_max - D_min + np.finfo(float).eps)
    P = np.exp(-D_scaled * beta_val)
    sumP = np.sum(P)
    H = np.log(sumP) + beta_val * np.sum(D_scaled * P) / sumP
    P = P / sumP
    return H, P

def umkl_bo(D, beta_val=None):
    if beta_val is None:
        beta_val = 1.0 / len(D)
    tol = 1e-4
    u = 150
    logU = np.log(u)
    H, thisP = Hbeta(D, beta_val)
    betamin = -np.inf
    betamax = np.inf
    Hdiff = H - logU
    tries = 0
    while abs(Hdiff) > tol and tries < 30:
        if Hdiff > 0:
            betamin = beta_val
            if np.isinf(betamax):
                beta_val = beta_val * 2
            else:
                beta_val = (beta_val + betamax) / 2.0
        else:
            betamax = beta_val
            if np.isinf(betamin):
                beta_val = beta_val / 2.0
            else:
                beta_val = (beta_val + betamin) / 2.0
        H, thisP = Hbeta(D, beta_val)
        Hdiff = H - logU
        tries += 1
    return thisP

def Kbeta(Ks, w):
    K = np.zeros((Ks.shape[0], Ks.shape[1]))
    for i in range(Ks.shape[2]):
        K += w[i] * Ks[:, :, i]
    return K

def tsne_p_bo(P, labels=None, no_dims=2):
    n = P.shape[0]
    momentum = 0.08
    final_momentum = 0.1
    mom_switch_iter = 250
    stop_lying_iter = 100
    max_iter = 1000
    epsilon = 500
    min_gain = 0.01
    
    np.fill_diagonal(P, 0)
    P = 0.5 * (P + P.T)
    P = np.maximum(P / np.sum(P), np.finfo(float).tiny)
    P = P * 4.0
    
    ydata = 0.0001 * np.random.randn(n, no_dims)
    y_incs = np.zeros_like(ydata)
    gains = np.ones_like(ydata)
    
    for iter in range(max_iter):
        sum_ydata = np.sum(ydata**2, axis=1)
        num = 1.0 / (1.0 + sum_ydata[:, None] + sum_ydata[None, :] - 2 * (ydata @ ydata.T))
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / np.sum(num), np.finfo(float).tiny)
        
        L = (P - Q) * num
        y_grads = 4 * (np.diag(np.sum(L, axis=0)) - L) @ ydata
        
        gains = (gains + 0.2) * (np.sign(y_grads) != np.sign(y_incs)) + (gains * 0.8) * (np.sign(y_grads) == np.sign(y_incs))
        gains = np.maximum(gains, min_gain)
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        ydata = ydata + y_incs
        ydata = ydata - np.mean(ydata, axis=0)
        ydata = np.clip(ydata, -100, 100)
        
        if iter == mom_switch_iter:
            momentum = final_momentum
        if iter == stop_lying_iter:
            P = P / 4.0
            
    return ydata

def litekmeans(X, c, start=None):
    if start is not None:
        kmeans = KMeans(n_clusters=c, init=start, n_init=1).fit(X)
    else:
        kmeans = KMeans(n_clusters=c, n_init=200).fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

def LaplacianScore(X, W):
    nSmp, nFea = X.shape
    D = np.sum(W, axis=1)
    L = W
    tmp1 = D.T @ X
    DPrime = np.sum((X.T * D).T * X, axis=0) - tmp1*tmp1/np.sum(D)
    LPrime = np.sum((X.T @ L).T * X, axis=0) - tmp1*tmp1/np.sum(D)
    DPrime[DPrime < 1e-12] = 10000
    Y = LPrime / DPrime
    return Y
