import numpy as np

def online_simple_regression(xs, ys):
    # return (N, 2) array of (w0, w1) after each point
    w_res = [[ys[0], 0]]
    for n in range(len(xs)):
        if n == 0:
            continue
        x = np.array(xs[:n+1])
        x_mean = np.mean(x)
        y = np.array(ys[:n+1])
        y_mean = np.mean(y)
        c_xx = np.sum((x - x_mean) ** 2) / (n+1)
        c_xy = np.sum((x - x_mean) * (y - y_mean)) / (n+1)
        if c_xx == 0:
            w1 = 0
        else:
            w1 = c_xy / c_xx
        w0 = y_mean - w1 * x_mean
        w_res.append([w0, w1])
    return np.array(w_res)
