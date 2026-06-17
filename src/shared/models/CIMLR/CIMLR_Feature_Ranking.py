import numpy as np
from scipy.stats import beta
from .utils import LaplacianScore

def betaScores(r):
    n = np.sum(~np.isnan(r))
    p = np.full(len(r), np.nan)
    r_sorted = np.sort(r)
    # p(1:n) = betacdf(r(1:n),1:n,n:-1:1)
    # In python: scipy.stats.beta.cdf(x, a, b)
    a = np.arange(1, n + 1)
    b = np.arange(n, 0, -1)
    p[:n] = beta.cdf(r_sorted[:n], a, b)
    return p

def correctBetaPvalues(p, k):
    return beta.cdf(p, 1, k)

def rhoScores(r):
    rows = r.shape[0]
    rho = np.full(rows, np.nan)
    for rInd in range(rows):
        r1 = r[rInd, :]
        x = betaScores(r1)
        k = np.sum(~np.isnan(x))
        rho[rInd] = correctBetaPvalues(np.min(x), k)
    return rho

def aggregateRanks(R):
    aggR = rhoScores(R)
    pval = aggR
    return aggR, pval

def CIMLR_Feature_Ranking(A, X):
    n_samples = len(A)
    n_features = X.shape[1]
    yscore = np.zeros((100, n_features))
    
    for i in range(100):
        index = np.random.permutation(n_samples)
        index = index[:int(round(n_samples * 0.9))]
        Ai = A[np.ix_(index, index)]
        Xi = X[index, :]
        yscore[i, :] = LaplacianScore(Xi, Ai)
        
    yscore = 1 - yscore
    glist = (yscore.T - np.min(yscore) + np.finfo(float).eps) / (np.max(yscore) - np.min(yscore) + np.finfo(float).eps)
    
    aggR, pval = aggregateRanks(glist)
    aggR_idx = np.argsort(pval)
    pval = pval[aggR_idx]
    
    return aggR_idx, pval
