import numpy as np

def forward_sub(L, b):
    y = np.array(b)
    for i in range(len(b)):
        for j in range(i):
            y[i] -= y[j] * L[i][j]
        y[i] /= L[i][i]
    return y

def back_sub(U, y):
    x = np.array(y)
    n = len(y)
    for rev_i in range(n):
        i = n - rev_i - 1
        for rev_j in range(rev_i):
            j = n - rev_j - 1
            x[i] -= x[j] * U[i][j]
        x[i] /= U[i][i]
    return x

def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    y = forward_sub(L, b)
    x = back_sub(L.T, y)
    return x

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    return 2*np.sum(np.log(np.diag(L)))

def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    return mu + eps @ L.T
