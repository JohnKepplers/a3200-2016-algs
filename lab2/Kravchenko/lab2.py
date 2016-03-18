import numpy as np


def linear_regression(x, y):
    m = x.shape[0]
    matrix = np.ndarray(shape=(m, 2))
    for j in range(m):
        matrix[j, 0] = 1
        matrix[j, 1] = x[j].tolist()[0]
    b = my_function(matrix, y)

    def f(t):
        p = b[0] + b[1] * t
        return p

    return f


def polynomial_regression(x, y, k):
    m = x.shape[0]
    matrix = np.ndarray(shape=(m, k + 1))
    for j in range(m):
        matrix[j, 0] = 1
        for z in range(1, k + 1):
            matrix[j, z] = x[j].tolist()[0] ** z
    b = my_function(matrix, y)

    def f(t):
        p = 0
        for i in range(k + 1):
            p += b[i] * (t ** i)
        return p

    return f


def functional_regression(x, y, functions):
    k = len(functions)
    m = x.shape[0]
    matrix = np.ndarray(shape=(m, k))
    for j in range(m):
        for z in range(k):
            matrix[j, z] = functions[z](x[j].tolist()[0])
    b = my_function(matrix, y)

    def f(t):
        p = 0
        for i in range(k):
            p += b[i] * functions[i](t)
        return p

    return f


def my_function(x, y):
    b = (x.T * x).I * x.T * y
    return b

