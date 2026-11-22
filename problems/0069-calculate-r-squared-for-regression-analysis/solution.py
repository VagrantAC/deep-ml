
import numpy as np

def r_squared(y_true, y_pred):
    ssr = np.sum((y_pred - y_true) ** 2)
    mean = np.mean(y_true)
    sst = np.sum((y_true - mean) ** 2)
    return 1 - (ssr / sst)