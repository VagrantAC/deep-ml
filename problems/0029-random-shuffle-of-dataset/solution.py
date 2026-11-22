import numpy as np

def shuffle_data(X, y, seed=None):
    # Your code here
    n = X.shape[0]
    if seed is not None:
        rng = np.random.seed(seed)
    idx = np.random.permutation(n)
    return X[idx], y[idx]
    pass