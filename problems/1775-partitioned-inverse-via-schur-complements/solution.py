import numpy as np

def partitioned_inverse(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv = np.linalg.inv(H)

    S = E - F @ H_inv @ G
    S_inv = np.linalg.inv(S)

    top_left = S_inv
    top_right = -S_inv @ F @ H_inv
    bottom_left = -H_inv @ G @ S_inv
    bottom_right = H_inv + H_inv @ G @ S_inv @ F @ H_inv

    return np.block([[top_left, top_right],
                     [bottom_left, bottom_right]])

def block_slogdet(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    # 这才是 M/H
    H_inv = np.linalg.inv(H)
    S = E - F @ H_inv @ G

    s1, l1 = np.linalg.slogdet(S)
    s2, l2 = np.linalg.slogdet(H)
    return (s1 * s2, l1 + l2)

def bordered_inverse(A_inv, b, c):
    v = A_inv @ b
    s = c - b @ v
    if abs(s) < 1e-12:
        raise ValueError("s too small")
    inv_s = 1.0 / s
    top_left = A_inv + inv_s * np.outer(v, v)
    top_right = -inv_s * v
    bottom_left = -inv_s * v
    bottom_right = np.array([[inv_s]])
    return np.block([[top_left, top_right[:,None]],
                     [bottom_left[None,:], bottom_right]])
