import numpy as np

def woodbury_solve(sigma_diag, X, B):
    # Σ^{-1} 就是逐元素倒数
    inv_sigma = 1.0 / sigma_diag  # (N,)
    # 小矩阵 M = I + X^T Σ^{-1} X, D×D
    XS = X * inv_sigma[:, None]   # 每行除以sigma，等价 Σ^{-1}X
    M = np.eye(X.shape[1]) + X.T @ XS
    # 解小系统，不求逆：先算 X^T Σ^{-1} B
    B_2d = B if B.ndim == 2 else B[:, None]
    tmp = X.T @ (B_2d * inv_sigma[:, None])  # D×K
    Y = np.linalg.solve(M, tmp)              # D×K
    res = B_2d * inv_sigma[:, None] - XS @ Y # N×K
    return res if B.ndim == 2 else res[:, 0]

def sherman_morrison(A_inv, u, v):
    denom = 1.0 + v @ (A_inv @ u)
    if denom == 0:
        raise ValueError("1+v^T A^{-1}u is zero, update is singular")
    Au = A_inv @ u
    vA = v @ A_inv
    return A_inv - np.outer(Au, vA) / denom

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    denom = 1.0 + v @ (A_inv @ u)
    if denom <= 0:
        raise ValueError("1+v^T A^{-1}u is zero, update is singular")

    return logdet_A + np.log(1 + v.T @ A_inv @ u)
