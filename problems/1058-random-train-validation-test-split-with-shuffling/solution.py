import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    
    rng = np.random.default_rng(seed)
    n = data.shape[0]

    indices = rng.permutation(n)

    train_end = int(train_frac * n)
    val_end = train_end + int(validation_frac * n)

    train_idx = indices[:train_end]
    if train_end != val_end:
        val_idx = indices[train_end:val_end]
    else:
        val_idx = np.array([], dtype=int)

    if val_end != n:
        test_idx = indices[val_end:n]
    else:
        test_idx = np.array([], dtype=int)

    return data[train_idx], data[val_idx], data[test_idx]