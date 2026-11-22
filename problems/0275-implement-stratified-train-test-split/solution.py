import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for c in classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)

        n_test = int(len(class_indices) * test_size)

        test_idx = class_indices[:n_test]
        train_idx = class_indices[n_test:]

        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return X_train, X_test, y_train, y_test
    
